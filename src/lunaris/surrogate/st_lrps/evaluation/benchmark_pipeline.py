# -*- coding: utf-8 -*-
"""Config-driven benchmark pipeline glue for ``lunaris-benchmark``."""

from __future__ import annotations

import csv
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .benchmark_config import load_benchmark_config
from .benchmark_validation import validate_benchmark_outputs
from .provenance import build_benchmark_manifest, sha256_payload, write_json
from lunaris.surrogate.st_lrps.artifacts.manager import read_artifact_contract
from lunaris.surrogate.st_lrps.shared.contracts import ArtifactContract, ArtifactContractError


def run_configured_benchmark(
    config_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    scenario_count: int | None = None,
    seed: int | None = None,
    dtype: str | None = None,
    quick: bool = False,
    allow_validation_fail: bool = False,
    allow_contract_mismatch: bool = False,
    allow_domain_extrapolation: bool = False,
    allow_legacy_artifact: bool = False,
) -> int:
    """Run a benchmark from a fixed config and write standardized outputs."""

    overrides = {
        "out_dir": out_dir,
        "model_dir": model_dir,
        "scenario_count": scenario_count,
        "seed": seed,
        "dtype": dtype,
        "quick": quick,
    }
    config = load_benchmark_config(config_path, overrides)
    output_dir = _resolve_output_dir(config, out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    config.setdefault("outputs", {})["out_dir"] = str(output_dir)

    contract_report = _benchmark_contract_report(
        config,
        strict=not bool(allow_contract_mismatch),
        strict_domain=not bool(allow_domain_extrapolation),
        allow_legacy_artifact=bool(allow_legacy_artifact),
    )
    config.setdefault("contract_compatibility", contract_report)
    resolved_hash = sha256_payload(config)
    manifest = build_benchmark_manifest(
        config=config,
        config_path=config_path,
        resolved_config_sha256=resolved_hash,
        output_dir=output_dir,
        cwd=Path.cwd(),
    )
    manifest["contract_compatibility"] = contract_report
    write_json(output_dir / "resolved_config.json", config)
    write_json(output_dir / "benchmark_manifest.json", manifest)

    _write_report(output_dir, config, validation_report=None, warnings=["Benchmark execution has not finished yet."])
    synthetic = bool(config.get("run_options", {}).get("synthetic", False))
    if synthetic:
        _write_synthetic_outputs(output_dir, config)
    else:
        _run_existing_harness(output_dir, config)
        _standardize_legacy_outputs(output_dir, config)

    validation_report = validate_benchmark_outputs(
        output_dir,
        resolved_config=config,
        expected_count=int(config["scenario"]["count"]),
        write_report=True,
    )
    _write_report(output_dir, config, validation_report=validation_report, warnings=validation_report["warnings"])

    if not validation_report["passed"] and not allow_validation_fail:
        for message in validation_report["errors"]:
            print(f"[validation] ERROR: {message}", flush=True)
        return 2
    if not validation_report["passed"]:
        print("[validation] Benchmark validation failed but --allow-validation-fail was set.", flush=True)
    else:
        print(f"[validation] Benchmark validation passed: {output_dir / 'validation_report.json'}", flush=True)
    return 0


def config_to_legacy_argv(config: Mapping[str, Any], output_dir: str | Path) -> list[str]:
    """Translate a resolved config into the existing harness flags."""

    scenario = config["scenario"]
    propagation = config["propagation"]
    truth = config["truth"]
    surrogate = config["surrogate"]
    baselines = config.get("baselines", [])
    gpu_models = [_baseline_model_token(item) for item in baselines if isinstance(item, Mapping)]
    if surrogate.get("enabled"):
        gpu_models.append("st_lrps")
    gpu_models = [m for i, m in enumerate(gpu_models) if m and m not in gpu_models[:i]]

    argv = [
        "--random-scenarios",
        str(int(scenario["count"])),
        "--scenario-seed",
        str(int(scenario["seed"])),
        "--scenario-mode",
        str(scenario["type"]),
        "--altitude-min-km",
        str(float(scenario["altitude_min_km"])),
        "--altitude-max-km",
        str(float(scenario["altitude_max_km"])),
        "--duration-days",
        str(float(propagation["duration_days"])),
        "--dt-out",
        str(float(propagation["output_dt_s"])),
        "--truth",
        f"sh{int(truth['degree'])}",
        "--truth-integrator",
        str(truth.get("integrator", "DOP853")),
        "--rtol",
        str(float(truth.get("rtol", 1.0e-10))),
        "--atol",
        str(float(truth.get("atol", 1.0e-12))),
        "--gpu-batch-compare",
        "--gpu-models",
        ",".join(gpu_models),
        "--gpu-integrator",
        _legacy_gpu_integrator(str(propagation.get("integrator", "RK4"))),
        "--rk4-dt-s",
        str(float(propagation["dt_s"])),
        "--torch-dtype",
        str(propagation["dtype"]),
        "--output-dir",
        str(output_dir),
    ]
    if _eccentricity_mode(config) == "circular_to_elliptic":
        argv.extend(["--ecc-min", "0.0", "--ecc-max", "0.2"])
    model_dir = surrogate.get("model_dir")
    if model_dir:
        argv.extend(["--st-lrps-model-dir", str(model_dir)])
    return argv


def _benchmark_contract_report(
    config: Mapping[str, Any],
    *,
    strict: bool,
    strict_domain: bool,
    allow_legacy_artifact: bool,
) -> dict[str, Any]:
    surrogate = config.get("surrogate", {}) if isinstance(config.get("surrogate"), Mapping) else {}
    if not surrogate.get("enabled"):
        return {"checked": False, "reason": "surrogate disabled"}
    model_dir = surrogate.get("model_dir")
    if not model_dir:
        return {
            "checked": False,
            "warnings": ["surrogate model_dir not configured; artifact contract cannot be checked"],
            "errors": [],
        }
    requested = ArtifactContract.from_benchmark_config(config)
    try:
        artifact = read_artifact_contract(
            model_dir,
            strict=True,
            allow_legacy_contract=bool(allow_legacy_artifact),
        )
        report = artifact.compatibility_report(requested, strict_domain=strict_domain)
        report["checked"] = True
        report["allow_legacy_artifact"] = bool(allow_legacy_artifact)
        if strict and report["errors"]:
            raise ArtifactContractError("; ".join(report["errors"]))
        if not strict and report["errors"]:
            report["warnings"] = list(report.get("warnings", []) or []) + [
                "contract mismatch allowed explicitly: " + str(message)
                for message in report.get("errors", []) or []
            ]
            report["errors"] = []
        return report
    except Exception as exc:
        if strict:
            raise
        return {
            "checked": True,
            "compatible": False,
            "errors": [],
            "warnings": [f"contract mismatch allowed explicitly: {exc}"],
            "allow_legacy_artifact": bool(allow_legacy_artifact),
        }


def _run_existing_harness(output_dir: Path, config: Mapping[str, Any]) -> None:
    from . import compare_gravity_models as cgm

    args = cgm.parse_args(config_to_legacy_argv(config, output_dir))
    cgm.run_from_args(args)


def _standardize_legacy_outputs(output_dir: Path, config: Mapping[str, Any]) -> None:
    metrics_dir = output_dir / "metrics"
    aggregate = metrics_dir / "gpu_batch_aggregate_metrics.csv"
    per_scenario = metrics_dir / "gpu_batch_per_scenario_metrics.csv"
    runtime = metrics_dir / "gpu_batch_runtime_metrics.csv"
    if not aggregate.exists():
        aggregate = output_dir / "aggregate_summary.csv"
    if not per_scenario.exists():
        per_scenario = output_dir / "per_scenario_metrics.csv"
    if not runtime.exists():
        runtime = output_dir / "batch_rk4_runtime_summary.csv"

    if aggregate.exists():
        shutil.copyfile(aggregate, output_dir / "metrics_summary.csv")
        rows = _read_csv(aggregate)
        write_json(
            output_dir / "metrics_summary.json",
            {"schema_version": 1, "units": _metric_units(), "rows": rows},
        )
    if per_scenario.exists():
        shutil.copyfile(per_scenario, output_dir / "scenario_results.csv")
    if runtime.exists():
        shutil.copyfile(runtime, output_dir / "runtime_summary.csv")


def _write_synthetic_outputs(output_dir: Path, config: Mapping[str, Any]) -> None:
    rng = np.random.default_rng(int(config["scenario"]["seed"]))
    scenario_count = int(config["scenario"]["count"])
    duration_days = float(config["propagation"]["duration_days"])
    dt_s = float(config["propagation"]["dt_s"])
    n_steps = max(1, int(math.ceil(duration_days * 86400.0 / max(dt_s, 1e-9))))
    models = _configured_model_names(config)
    scenario_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for model_index, model in enumerate(models):
        model_factor = 1.0 + 0.35 * model_index
        runtime = max(0.001, 0.02 * scenario_count * model_factor)
        runtime_rows.append(
            {
                "model": model,
                "backend": "synthetic",
                "device": "cpu",
                "dtype": config["propagation"]["dtype"],
                "n_scenarios": scenario_count,
                "n_steps": n_steps * scenario_count,
                "total_runtime_s": runtime,
                "runtime_per_scenario_s": runtime / scenario_count,
                "status": "synthetic",
            }
        )
        for scenario_id in range(scenario_count):
            base = (scenario_id + 1) * 0.001 * model_factor
            jitter = float(rng.uniform(0.0, 0.0002))
            rms = base + jitter
            scenario_rows.append(
                {
                    "scenario_id": scenario_id,
                    "model": model,
                    "reference": f"SH{int(config['truth']['degree'])}",
                    "rms_pos_err_km": rms,
                    "final_pos_err_km": rms * 1.1,
                    "max_pos_err_km": rms * 1.5,
                    "p95_pos_err_km": rms * 1.35,
                    "rms_vel_err_ms": rms * 0.1,
                    "final_vel_err_ms": rms * 0.12,
                    "radial_rms_km": rms * 0.05,
                    "along_rms_km": rms * 0.9,
                    "cross_rms_km": rms * 0.03,
                    "rms_alt_err_km": rms * 0.04,
                    "runtime_s": runtime / scenario_count,
                    "n_steps": n_steps,
                    "status": "ok",
                    "domain_warning": "",
                    "distance_unit": "km",
                    "time_unit": "s",
                }
            )

    summary_rows = _aggregate_synthetic_metrics(scenario_rows)
    _write_csv(output_dir / "scenario_results.csv", scenario_rows)
    _write_csv(output_dir / "metrics_summary.csv", summary_rows)
    _write_csv(output_dir / "runtime_summary.csv", runtime_rows)
    write_json(
        output_dir / "metrics_summary.json",
        {
            "schema_version": 1,
            "units": _metric_units(),
            "rows": summary_rows,
            "synthetic": True,
        },
    )


def _aggregate_synthetic_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    out = []
    for model, model_rows in grouped.items():
        rms = np.asarray([float(row["rms_pos_err_km"]) for row in model_rows], dtype=float)
        vel = np.asarray([float(row["rms_vel_err_ms"]) for row in model_rows], dtype=float)
        radial = np.asarray([float(row["radial_rms_km"]) for row in model_rows], dtype=float)
        along = np.asarray([float(row["along_rms_km"]) for row in model_rows], dtype=float)
        cross = np.asarray([float(row["cross_rms_km"]) for row in model_rows], dtype=float)
        out.append(
            {
                "model": model,
                "n_scenarios_ok": len(model_rows),
                "n_scenarios_failed": 0,
                "mean_rms_pos_err_km": float(np.mean(rms)),
                "median_rms_pos_err_km": float(np.median(rms)),
                "p95_rms_pos_err_km": float(np.percentile(rms, 95)),
                "max_rms_pos_err_km": float(np.max(rms)),
                "median_rms_vel_err_ms": float(np.median(vel)),
                "p95_rms_vel_err_ms": float(np.percentile(vel, 95)),
                "max_rms_vel_err_ms": float(np.max(vel)),
                "median_radial_rms_km": float(np.median(radial)),
                "median_along_rms_km": float(np.median(along)),
                "median_cross_rms_km": float(np.median(cross)),
            }
        )
    return out


def _write_report(
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    validation_report: Mapping[str, Any] | None,
    warnings: list[str],
) -> None:
    scenario = config["scenario"]
    truth = config["truth"]
    models = ", ".join(_configured_model_names(config))
    status = "pending" if validation_report is None else ("passed" if validation_report.get("passed") else "failed")
    lines = [
        f"# Benchmark Report: {config['name']}",
        "",
        f"- Benchmark name: {config['name']}",
        f"- Scenario count: {scenario['count']}",
        f"- Duration days: {config['propagation']['duration_days']}",
        f"- Truth model: {truth['model']} degree {truth['degree']}",
        f"- Compared models: {models}",
        f"- Validation status: {status}",
        f"- Metrics CSV: {output_dir / 'metrics_summary.csv'}",
        f"- Scenario CSV: {output_dir / 'scenario_results.csv'}",
        f"- Runtime CSV: {output_dir / 'runtime_summary.csv'}",
        "",
        "## Warnings",
    ]
    lines.extend([f"- {warning}" for warning in warnings] or ["- None"])
    if validation_report is not None and validation_report.get("errors"):
        lines.extend(["", "## Validation Errors"])
        lines.extend(f"- {message}" for message in validation_report["errors"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_output_dir(config: Mapping[str, Any], override: str | Path | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    configured = config.get("outputs", {}).get("out_dir") if isinstance(config.get("outputs"), Mapping) else None
    if configured:
        return Path(configured).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (Path("outputs") / "gravity_benchmark" / f"{config['name']}_{timestamp}").resolve()


def _configured_model_names(config: Mapping[str, Any]) -> list[str]:
    names = [str(item.get("name")) for item in config.get("baselines", []) if isinstance(item, Mapping)]
    surrogate = config.get("surrogate", {})
    if isinstance(surrogate, Mapping) and surrogate.get("enabled"):
        names.append(str(surrogate.get("name", "ST-LRPS")))
    return [name for i, name in enumerate(names) if name and name not in names[:i]]


def _baseline_model_token(item: Mapping[str, Any]) -> str:
    if item.get("model") == "spherical_harmonics":
        return f"sh{int(item['degree'])}"
    return str(item.get("name", "")).lower()


def _legacy_gpu_integrator(value: str) -> str:
    if value.upper() == "RK4":
        return "medium"
    return "medium"


def _eccentricity_mode(config: Mapping[str, Any]) -> str:
    scenario = config.get("scenario", {})
    if isinstance(scenario, Mapping):
        return str(scenario.get("eccentricity_mode", ""))
    return ""


def _metric_units() -> dict[str, str]:
    return {
        "distance": "km",
        "velocity": "m/s",
        "time": "s",
        "runtime": "s",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
