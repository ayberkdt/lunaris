"""Config-driven benchmark pipeline glue for ``lunaris-benchmark``."""

from __future__ import annotations

import csv
import math
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.common.constants import DAY_S
from lunaris.common.provenance import utc_now_iso
from lunaris.surrogate.st_lrps.shared.contracts import ArtifactContract, ArtifactContractError

from .benchmark_config import (
    SYNTHETIC_BANNER,
    apply_paper_safe,
    is_paper_safe_requested,
    load_benchmark_config,
)
from .benchmark_validation import validate_benchmark_outputs
from .provenance import build_benchmark_manifest, sha256_payload, write_json

# This pipeline compares gravity models only (classical SH vs ST-LRPS); no
# non-gravity perturbation is ever enabled here. Results produced with a
# full-dynamics force set belong in a separate table and must not be merged
# with rows carrying this scope.
FORCE_MODEL_SCOPE = "gravity_only"


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
    paper_safe: bool = False,
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

    # Paper-safe mode hard-fails on synthetic/quick/mismatch/extrapolation
    # settings *before* any output is produced, and forces the strict flags so a
    # debug/legacy benchmark can never masquerade as a scientific result.
    paper_safe = is_paper_safe_requested(config, flag=bool(paper_safe))
    paper_safe_enforced: dict[str, Any] | None = None
    if paper_safe:
        paper_safe_enforced = apply_paper_safe(config)
        allow_validation_fail = False
        allow_contract_mismatch = False
        allow_domain_extrapolation = False

    output_dir = _resolve_output_dir(config, out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    config.setdefault("outputs", {})["out_dir"] = str(output_dir)

    contract_report = _benchmark_contract_report(
        config,
        strict=not bool(allow_contract_mismatch),
        strict_domain=not bool(allow_domain_extrapolation),
        paper_safe=bool(paper_safe),
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
    manifest["paper_safe"] = {"enabled": bool(paper_safe), "enforced": paper_safe_enforced}
    # A crashed run must be distinguishable from a validated one: the manifest
    # says "pending" until validation actually finishes below.
    manifest["validation"] = {"status": "pending"}
    # Always record the exact invocation so a benchmark can be reproduced.
    _write_run_command(output_dir)
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

    evidence = validation_report.get("evidence", {})
    manifest["validation"] = {
        "status": "passed" if validation_report["passed"] else "failed",
        "passed": bool(validation_report["passed"]),
        "error_count": len(validation_report["errors"]),
        "warning_count": len(validation_report["warnings"]),
        "scientific_evidence": bool(evidence.get("scientific_evidence", False)),
        "allow_validation_fail": bool(allow_validation_fail),
        "validated_at_utc": utc_now_iso(),
        "report_path": "validation_report.json",
    }
    # Surface the error-decomposition contract in the manifest itself so a
    # reviewer sees the field/orbit/integrator/phase/runtime split and the
    # backend/dtype/frame provenance without opening validation_report.json.
    if isinstance(validation_report.get("error_decomposition"), Mapping):
        manifest["error_decomposition"] = validation_report["error_decomposition"]
    write_json(output_dir / "benchmark_manifest.json", manifest)

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
    if surrogate.get("enabled") and surrogate.get("cpu_adaptive"):
        argv.append("--cpu-adaptive-surrogate")
        if surrogate.get("cpu_adaptive_rtol") is not None:
            argv.extend(["--cpu-adaptive-rtol", str(float(surrogate["cpu_adaptive_rtol"]))])
        if surrogate.get("cpu_adaptive_atol") is not None:
            argv.extend(["--cpu-adaptive-atol", str(float(surrogate["cpu_adaptive_atol"]))])
    if surrogate.get("enabled") and surrogate.get("require_st_lrps"):
        # Fail closed: a benchmark whose scientific point is the surrogate must
        # refuse to run (rather than silently degrade to an SH-only ladder)
        # when no valid model directory can be resolved.
        argv.append("--require-st-lrps")
    return argv


def _benchmark_contract_report(
    config: Mapping[str, Any],
    *,
    strict: bool,
    strict_domain: bool,
    paper_safe: bool = False,
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
        from lunaris.surrogate.st_lrps.artifacts.manager import (
            paper_safe_metadata_report_for_run,
            read_artifact_contract,
        )

        artifact = read_artifact_contract(
            model_dir,
            strict=True,
        )
        report = artifact.compatibility_report(requested, strict_domain=strict_domain)
        report["checked"] = True
        if paper_safe:
            # R26: a paper-safe benchmark refuses to start when the artifact's
            # required metadata is incomplete. The completeness report lands in
            # the manifest via contract_compatibility for provenance.
            ps_report = paper_safe_metadata_report_for_run(model_dir)
            report["paper_safe_metadata"] = ps_report
            if not ps_report["complete"]:
                raise ArtifactContractError(
                    "paper_safe benchmark refused: surrogate artifact metadata is "
                    f"incomplete (missing: {', '.join(ps_report['missing'])}). "
                    "Regenerate the artifact with the current training pipeline."
                )
            # R29b (#6): gravity-file identity. The artifact records the SHA-256
            # of the gravity model its labels came from; a paper-safe benchmark
            # against a *different* gravity file is comparing apples to oranges.
            identity = _gravity_file_identity_check(config, ps_report["fields"])
            report["gravity_file_identity"] = identity
            if identity["status"] == "mismatch":
                raise ArtifactContractError(
                    "paper_safe benchmark refused: configured truth gravity file "
                    f"{identity['configured_path']} (sha256={identity['configured_sha256']}) "
                    "does not match the artifact's training gravity model "
                    f"(sha256={identity['artifact_sha256']})."
                )
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
        }


def _gravity_file_identity_check(
    config: Mapping[str, Any],
    metadata_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """R29b (#6): compare the configured truth gravity file with the artifact's.

    Returns a JSON-serializable status block. ``status`` is one of:
    ``match`` / ``mismatch`` / ``unverified`` (no truth.gravity_file configured,
    or the file cannot be hashed). Only ``mismatch`` is fatal under paper-safe;
    ``unverified`` is recorded so the gap is visible in the manifest.
    """
    artifact_sha = metadata_fields.get("gravity_model_hash")
    truth = config.get("truth") if isinstance(config.get("truth"), Mapping) else {}
    gravity_file = truth.get("gravity_file")
    if not gravity_file:
        return {
            "status": "unverified",
            "reason": "truth.gravity_file not configured; artifact hash recorded only",
            "configured_path": None,
            "configured_sha256": None,
            "artifact_sha256": artifact_sha,
        }
    path = Path(str(gravity_file))
    if not path.exists():
        return {
            "status": "unverified",
            "reason": f"truth.gravity_file does not exist: {path}",
            "configured_path": str(path),
            "configured_sha256": None,
            "artifact_sha256": artifact_sha,
        }
    from lunaris.common.provenance import sha256_file

    configured_sha = str(sha256_file(path))
    return {
        "status": "match" if configured_sha == str(artifact_sha) else "mismatch",
        "reason": None,
        "configured_path": str(path),
        "configured_sha256": configured_sha,
        "artifact_sha256": artifact_sha,
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
            {
                "schema_version": 1,
                "units": _metric_units(),
                "force_model_scope": FORCE_MODEL_SCOPE,
                "rows": rows,
            },
        )
    if per_scenario.exists():
        rows = _standardize_scenario_rows(_read_csv(per_scenario), synthetic=False)
        _write_csv(output_dir / "scenario_results.csv", rows)
    if runtime.exists():
        rows = _standardize_runtime_rows(_read_csv(runtime), config, synthetic=False)
        _write_csv(output_dir / "runtime_summary.csv", rows)


def _write_synthetic_outputs(output_dir: Path, config: Mapping[str, Any]) -> None:
    rng = np.random.default_rng(int(config["scenario"]["seed"]))
    scenario_count = int(config["scenario"]["count"])
    duration_days = float(config["propagation"]["duration_days"])
    dt_s = float(config["propagation"]["dt_s"])
    n_steps = max(1, int(math.ceil(duration_days * DAY_S / max(dt_s, 1e-9))))
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
                    "cold_time_s": runtime + 0.002,
                    "warm_time_s": runtime,
                    "jit_compile_time_s": 0.002,
                    "propagation_time_s": runtime,
                    "acceleration_evaluations_per_second": (
                        scenario_count * n_steps * 4 / max(runtime, 1e-9)
                    ),
                    "propagated_seconds_per_wall_second": (
                        scenario_count * duration_days * DAY_S / max(runtime, 1e-9)
                    ),
                    "peak_memory_mb": 0.0,
                    "chunk_size": scenario_count,
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
                    "phase_lag_final_s": rms * 0.4 * (1.0 if scenario_id % 2 else -1.0),
                    "phase_lag_slope_s_per_day": rms * 0.08,
                    "phase_corrected_rms_km": rms * 0.12,
                    "phase_explained_fraction": 0.85,
                    "rms_alt_err_km": rms * 0.04,
                    "trajectory_rms_km": rms,
                    "energy_drift_rel": rms * 1.0e-8,
                    "accel_max_error_m_s2": rms * 1.0e-9,
                    "potential_error_m2_s2": rms * 1.0e-3,
                    "impact_count": 0,
                    "domain_exit_count": 0,
                    "extended_metrics_source": "synthetic_smoke",
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
            "force_model_scope": FORCE_MODEL_SCOPE,
            "rows": summary_rows,
            "synthetic": True,
            "warning": SYNTHETIC_BANNER,
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
        abs_lag = np.asarray(
            [abs(float(row["phase_lag_final_s"])) for row in model_rows], dtype=float)
        pcr = np.asarray(
            [float(row["phase_corrected_rms_km"]) for row in model_rows], dtype=float)
        pef = np.asarray(
            [float(row["phase_explained_fraction"]) for row in model_rows], dtype=float)
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
                "median_abs_phase_lag_final_s": float(np.median(abs_lag)),
                "median_phase_corrected_rms_km": float(np.median(pcr)),
                "median_phase_explained_fraction": float(np.median(pef)),
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
    run_options = config.get("run_options", {}) if isinstance(config.get("run_options"), Mapping) else {}
    is_synthetic = bool(run_options.get("synthetic", False))
    lines = [
        f"# Benchmark Report: {config['name']}",
        "",
    ]
    if is_synthetic:
        lines += [f"> **{SYNTHETIC_BANNER}**", ""]
    if config.get("paper_safe"):
        lines += ["> Paper-safe mode: strict contract/domain enforcement, no synthetic/legacy settings.", ""]
    lines += [
        f"- Benchmark name: {config['name']}",
        f"- Scenario count: {scenario['count']}",
        f"- Duration days: {config['propagation']['duration_days']}",
        f"- Truth model: {truth['model']} degree {truth['degree']}",
        f"- Force-model scope: {FORCE_MODEL_SCOPE} (lunar gravity models only; "
        "no third-body, SRP, albedo, thermal IR, tides, or relativity). "
        "Gravity-only and full-dynamics results must never share a table.",
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
    timestamp = utc_now_iso().replace("-", "").replace(":", "")
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
    # The legacy harness exposes a single fixed-step integrator profile
    # ("medium"); the actual RK4 step size is carried separately via --rk4-dt-s.
    del value
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
        "acceleration": "m/s^2",
        "potential": "m^2/s^2",
        "energy_drift": "relative",
    }


def _standardize_scenario_rows(
    rows: list[dict[str, Any]],
    *,
    synthetic: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    source = "synthetic_smoke" if synthetic else "not_evaluated"
    for row in rows:
        item = dict(row)
        item.setdefault("trajectory_rms_km", item.get("rms_pos_err_km", ""))
        item.setdefault("final_pos_err_km", item.get("rms_pos_err_km", ""))
        item.setdefault("max_pos_err_km", item.get("rms_pos_err_km", ""))
        item.setdefault("p95_pos_err_km", item.get("rms_pos_err_km", ""))
        item.setdefault("rms_vel_err_ms", item.get("final_vel_err_ms", ""))
        item.setdefault("final_vel_err_ms", item.get("rms_vel_err_ms", ""))
        item.setdefault("energy_drift_rel", "" if not synthetic else 0.0)
        item.setdefault("accel_max_error_m_s2", "" if not synthetic else 0.0)
        item.setdefault("potential_error_m2_s2", "" if not synthetic else 0.0)
        item.setdefault("impact_count", "" if not synthetic else 0)
        item.setdefault("domain_exit_count", "" if not synthetic else 0)
        item.setdefault("extended_metrics_source", source)
        out.append(item)
    return out


def _standardize_runtime_rows(
    rows: list[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    synthetic: bool,
) -> list[dict[str, Any]]:
    scenario_count = int(config.get("scenario", {}).get("count", 1))
    duration_days = float(config.get("propagation", {}).get("duration_days", 0.0))
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        total = _float_or_default(item.get("total_runtime_s"), 0.0)
        warm = _float_or_default(item.get("warm_time_s"), total)
        jit = _float_or_default(item.get("jit_compile_time_s"), 0.0 if not synthetic else 0.002)
        cold = _float_or_default(item.get("cold_time_s"), warm + jit)
        steps = _float_or_default(item.get("n_steps"), 0.0)
        n_scenarios = _float_or_default(item.get("n_scenarios"), scenario_count)
        item.setdefault("cold_time_s", cold)
        item.setdefault("warm_time_s", warm)
        item.setdefault("jit_compile_time_s", jit)
        item.setdefault("propagation_time_s", _float_or_default(item.get("propagation_time_s"), warm))
        item.setdefault(
            "acceleration_evaluations_per_second",
            n_scenarios * max(steps, 0.0) * 4.0 / max(warm, 1.0e-9),
        )
        item.setdefault(
            "propagated_seconds_per_wall_second",
            n_scenarios * duration_days * DAY_S / max(warm, 1.0e-9),
        )
        item.setdefault("peak_memory_mb", "")
        item.setdefault("chunk_size", "")
        out.append(item)
    return out


def _float_or_default(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


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


def _write_run_command(output_dir: Path) -> None:
    """Persist the exact CLI invocation for reproducibility/provenance."""
    try:
        command = " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else ""
    except Exception:
        command = ""
    (output_dir / "run_command.txt").write_text(command + "\n", encoding="utf-8")
