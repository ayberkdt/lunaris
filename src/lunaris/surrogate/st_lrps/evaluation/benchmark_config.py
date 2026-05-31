# -*- coding: utf-8 -*-
"""Config loading for reproducible ST-LRPS benchmark runs."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping


SUPPORTED_DTYPES = {"float32", "float64"}
SUPPORTED_SCENARIO_TYPES = {"bounded_keplerian", "near_circular_altitude"}
SUPPORTED_TRUTH_MODELS = {"spherical_harmonics"}


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark config is ambiguous or invalid."""


def canonical_json_text(payload: Mapping[str, Any]) -> str:
    """Return stable JSON text used for resolved-config hashes."""

    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n"


def load_benchmark_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load, validate, and resolve a benchmark config file.

    JSON is supported without optional dependencies. YAML is supported when
    PyYAML is installed; otherwise a small strict parser handles the simple
    mapping/list/scalar subset used by Lunaris benchmark configs.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise BenchmarkConfigError(f"benchmark config does not exist: {config_path}")
    raw = _load_config_payload(config_path)
    if not isinstance(raw, dict):
        raise BenchmarkConfigError("benchmark config must be a mapping/object")

    resolved = copy.deepcopy(raw)
    _fill_safe_defaults(resolved)
    apply_benchmark_overrides(resolved, overrides or {})
    _normalize_paths(resolved, config_path)
    validate_benchmark_config(resolved)
    return resolved


def apply_benchmark_overrides(
    config: MutableMapping[str, Any],
    overrides: Mapping[str, Any],
) -> None:
    """Apply CLI overrides to the intended config fields only."""

    if _has_value(overrides.get("out_dir")):
        _ensure_mapping(config, "outputs")["out_dir"] = str(overrides["out_dir"])
    if _has_value(overrides.get("model_dir")):
        _ensure_mapping(config, "surrogate")["model_dir"] = str(overrides["model_dir"])
    if _has_value(overrides.get("scenario_count")):
        _ensure_mapping(config, "scenario")["count"] = int(overrides["scenario_count"])
    if _has_value(overrides.get("seed")):
        _ensure_mapping(config, "scenario")["seed"] = int(overrides["seed"])
    if _has_value(overrides.get("dtype")):
        _ensure_mapping(config, "propagation")["dtype"] = str(overrides["dtype"])
    if bool(overrides.get("quick")):
        apply_quick_mode(config)


def apply_quick_mode(config: MutableMapping[str, Any]) -> None:
    """Reduce a config enough to exercise the full pipeline in CI."""

    scenario = _ensure_mapping(config, "scenario")
    propagation = _ensure_mapping(config, "propagation")
    truth = _ensure_mapping(config, "truth")
    baselines = config.get("baselines")

    scenario["count"] = min(int(scenario.get("count", 3)), 3)
    propagation["duration_days"] = min(float(propagation.get("duration_days", 0.01)), 0.01)
    propagation["output_dt_s"] = max(float(propagation.get("output_dt_s", 60.0)), 60.0)
    propagation["dt_s"] = max(float(propagation.get("dt_s", 30.0)), 30.0)
    if str(truth.get("model")) == "spherical_harmonics" and "degree" in truth:
        truth["degree"] = min(int(truth["degree"]), 20)
    if isinstance(baselines, list):
        seen: set[str] = set()
        quick_baselines = []
        for item in baselines:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            if str(entry.get("model")) == "spherical_harmonics" and "degree" in entry:
                entry["degree"] = min(int(entry["degree"]), 20)
                entry["name"] = f"SH{entry['degree']}"
            name = str(entry.get("name", "")).strip()
            if name and name not in seen:
                quick_baselines.append(entry)
                seen.add(name)
        config["baselines"] = quick_baselines[:1] or [{"name": "SH20", "model": "spherical_harmonics", "degree": 20}]

    run_options = _ensure_mapping(config, "run_options")
    run_options["quick"] = True
    run_options["synthetic"] = True
    run_options["quick_note"] = "Reduced scenario count/duration and uses synthetic outputs for CI."
    validation = _ensure_mapping(config, "validation")
    validation["allow_truth_baseline"] = True


def validate_benchmark_config(config: Mapping[str, Any]) -> None:
    """Validate required fields and reject unsafe ambiguity."""

    _require(config, "schema_version")
    if int(config["schema_version"]) != 1:
        raise BenchmarkConfigError("schema_version must be 1")

    name = _required_str(config, "name")
    if not re.match(r"^[A-Za-z0-9_.-]+$", name):
        raise BenchmarkConfigError("name may contain only letters, numbers, '.', '_' and '-'")
    _required_str(config, "description")

    scenario = _required_mapping(config, "scenario")
    _require(scenario, "seed", "scenario.seed")
    _require(scenario, "count", "scenario.count")
    if int(scenario["count"]) <= 0:
        raise BenchmarkConfigError("scenario.count must be positive")
    scenario_type = _required_str(scenario, "type", "scenario.type")
    if scenario_type not in SUPPORTED_SCENARIO_TYPES:
        raise BenchmarkConfigError(f"unsupported scenario.type: {scenario_type}")
    alt_min = float(_require(scenario, "altitude_min_km", "scenario.altitude_min_km"))
    alt_max = float(_require(scenario, "altitude_max_km", "scenario.altitude_max_km"))
    if alt_min < 0 or alt_max <= alt_min:
        raise BenchmarkConfigError("scenario altitude bounds must satisfy 0 <= min < max")

    propagation = _required_mapping(config, "propagation")
    if float(_require(propagation, "duration_days", "propagation.duration_days")) <= 0:
        raise BenchmarkConfigError("propagation.duration_days must be positive")
    if float(_require(propagation, "output_dt_s", "propagation.output_dt_s")) <= 0:
        raise BenchmarkConfigError("propagation.output_dt_s must be positive")
    if float(_require(propagation, "dt_s", "propagation.dt_s")) <= 0:
        raise BenchmarkConfigError("propagation.dt_s must be positive")
    _required_str(propagation, "integrator", "propagation.integrator")
    dtype = _required_str(propagation, "dtype", "propagation.dtype")
    if dtype not in SUPPORTED_DTYPES:
        raise BenchmarkConfigError(f"unsupported propagation.dtype: {dtype}")

    truth = _required_mapping(config, "truth")
    truth_model = _required_str(truth, "model", "truth.model")
    if truth_model not in SUPPORTED_TRUTH_MODELS:
        raise BenchmarkConfigError(f"unsupported truth.model: {truth_model}")
    if int(_require(truth, "degree", "truth.degree")) <= 0:
        raise BenchmarkConfigError("truth.degree must be positive")

    baselines = config.get("baselines")
    if not isinstance(baselines, list) or not baselines:
        raise BenchmarkConfigError("baselines must be a non-empty list")
    names: set[str] = set()
    for index, baseline in enumerate(baselines):
        if not isinstance(baseline, dict):
            raise BenchmarkConfigError(f"baselines[{index}] must be a mapping")
        bname = _required_str(baseline, "name", f"baselines[{index}].name")
        if bname in names:
            raise BenchmarkConfigError(f"duplicate baseline name: {bname}")
        names.add(bname)
        bmodel = _required_str(baseline, "model", f"baselines[{index}].model")
        if bmodel not in SUPPORTED_TRUTH_MODELS:
            raise BenchmarkConfigError(f"unsupported baseline model: {bmodel}")
        if int(_require(baseline, "degree", f"baselines[{index}].degree")) <= 0:
            raise BenchmarkConfigError(f"baselines[{index}].degree must be positive")

    surrogate = _required_mapping(config, "surrogate")
    _require(surrogate, "enabled", "surrogate.enabled")
    _required_str(surrogate, "name", "surrogate.name")
    if int(_require(surrogate, "baseline_degree", "surrogate.baseline_degree")) <= 0:
        raise BenchmarkConfigError("surrogate.baseline_degree must be positive")

    outputs = _required_mapping(config, "outputs")
    for key in ("write_figures", "write_csv", "write_json"):
        if key not in outputs:
            raise BenchmarkConfigError(f"outputs.{key} is required")


def _load_config_payload(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json" or text.lstrip().startswith("{"):
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        return _load_yaml_payload(text)
    raise BenchmarkConfigError(f"unsupported config extension: {path.suffix}")


def _load_yaml_payload(text: str) -> Any:
    try:
        import yaml  # type: ignore
    except Exception:
        return _parse_simple_yaml(text)
    data = yaml.safe_load(text)
    return data


def _parse_simple_yaml(text: str) -> Any:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip()))
    if not lines:
        return {}
    value, index = _parse_yaml_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise BenchmarkConfigError("could not parse YAML config")
    return value


def _parse_yaml_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    mapping: dict[str, Any] = {}
    sequence: list[Any] = []
    mode: str | None = None
    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise BenchmarkConfigError(f"unexpected YAML indentation near: {text}")
        if text.startswith("- "):
            if mode == "map":
                raise BenchmarkConfigError("cannot mix YAML mapping and sequence at one indentation level")
            mode = "seq"
            item_text = text[2:].strip()
            if not item_text:
                item, index = _parse_yaml_block(lines, index + 1, indent + 2)
                sequence.append(item)
                continue
            if ":" in item_text:
                key, value_text = _split_yaml_key_value(item_text)
                item: dict[str, Any] = {}
                if value_text:
                    item[key] = _parse_yaml_scalar(value_text)
                    index += 1
                else:
                    item[key], index = _parse_yaml_block(lines, index + 1, indent + 2)
                if index < len(lines) and lines[index][0] == indent + 2 and not lines[index][1].startswith("- "):
                    child, index = _parse_yaml_block(lines, index, indent + 2)
                    if not isinstance(child, dict):
                        raise BenchmarkConfigError("YAML list item continuation must be a mapping")
                    item.update(child)
                sequence.append(item)
            else:
                sequence.append(_parse_yaml_scalar(item_text))
                index += 1
        else:
            if mode == "seq":
                raise BenchmarkConfigError("cannot mix YAML sequence and mapping at one indentation level")
            mode = "map"
            key, value_text = _split_yaml_key_value(text)
            if value_text:
                mapping[key] = _parse_yaml_scalar(value_text)
                index += 1
            else:
                mapping[key], index = _parse_yaml_block(lines, index + 1, indent + 2)
    return (sequence if mode == "seq" else mapping), index


def _split_yaml_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise BenchmarkConfigError(f"expected YAML key/value pair near: {text}")
    key, value = text.split(":", 1)
    key = key.strip()
    if not key:
        raise BenchmarkConfigError("YAML key cannot be empty")
    return key, value.strip()


def _parse_yaml_scalar(text: str) -> Any:
    value = text.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") or value.startswith("{"):
        return json.loads(value)
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _fill_safe_defaults(config: MutableMapping[str, Any]) -> None:
    outputs = _ensure_mapping(config, "outputs")
    outputs.setdefault("out_dir", None)
    outputs.setdefault("write_figures", True)
    outputs.setdefault("write_csv", True)
    outputs.setdefault("write_json", True)
    scenario = _ensure_mapping(config, "scenario")
    scenario.setdefault("eccentricity_mode", "circular_to_elliptic")
    scenario.setdefault("inclination_mode", "full_range")
    propagation = _ensure_mapping(config, "propagation")
    propagation.setdefault("integrator", "RK4")
    propagation.setdefault("dtype", "float64")
    surrogate = _ensure_mapping(config, "surrogate")
    surrogate.setdefault("enabled", False)
    surrogate.setdefault("name", "ST-LRPS")
    surrogate.setdefault("model_dir", None)
    surrogate.setdefault("baseline_degree", 20)


def _normalize_paths(config: MutableMapping[str, Any], config_path: Path) -> None:
    base = config_path.parent
    parts = list(config_path.parts)
    if "configs" in parts:
        idx = parts.index("configs")
        if idx > 0:
            base = Path(*parts[:idx])
    outputs = _ensure_mapping(config, "outputs")
    out_dir = outputs.get("out_dir")
    if out_dir:
        outputs["out_dir"] = str(_resolve_path(out_dir, base))
    surrogate = _ensure_mapping(config, "surrogate")
    model_dir = surrogate.get("model_dir")
    if model_dir:
        surrogate["model_dir"] = str(_resolve_path(model_dir, base))
    for section_name in ("truth", "dataset"):
        section = config.get(section_name)
        if isinstance(section, dict):
            for key in ("gravity_file", "file", "path"):
                if section.get(key):
                    section[key] = str(_resolve_path(section[key], base))


def _resolve_path(value: Any, base: Path) -> Path:
    expanded = os.path.expandvars(str(value))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


def _ensure_mapping(config: MutableMapping[str, Any], key: str) -> MutableMapping[str, Any]:
    value = config.get(key)
    if value is None:
        value = {}
        config[key] = value
    if not isinstance(value, MutableMapping):
        raise BenchmarkConfigError(f"{key} must be a mapping")
    return value


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _require(config, key)
    if not isinstance(value, Mapping):
        raise BenchmarkConfigError(f"{key} must be a mapping")
    return value


def _required_str(config: Mapping[str, Any], key: str, label: str | None = None) -> str:
    value = _require(config, key, label)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkConfigError(f"{label or key} must be a non-empty string")
    return value.strip()


def _require(config: Mapping[str, Any], key: str, label: str | None = None) -> Any:
    if key not in config or config[key] is None:
        raise BenchmarkConfigError(f"{label or key} is required")
    return config[key]
