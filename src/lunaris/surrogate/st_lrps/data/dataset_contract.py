# -*- coding: utf-8 -*-
"""Versioned ST-LRPS dataset contract utilities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI


DATASET_CONTRACT_SCHEMA_VERSION = 1
REQUIRED_DERIVATIVE_CONVENTION = "dP_dphi_corrected_v1"
TARGET_MODES = frozenset({"residual", "full"})
BASELINE_KINDS = frozenset({"none", "point_mass", "spherical_harmonics"})
DEFAULT_COORDINATE_FRAME = "moon_fixed_cartesian"
DEFAULT_UNITS = {"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"}
DATASET_CONTRACT_ATTR = "dataset_contract_json"
METADATA_GROUP = "metadata"


class DatasetContractError(ValueError):
    """Raised when dataset metadata is missing, ambiguous, or unsafe."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path | None) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256_for_hdf5_dataset(path: str | Path, dataset_name: str = "data", *, chunk_rows: int = 65536) -> str:
    import h5py  # type: ignore

    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        ds = handle[dataset_name]
        for start in range(0, int(ds.shape[0]), int(chunk_rows)):
            arr = np.asarray(ds[start : start + int(chunk_rows)])
            digest.update(np.ascontiguousarray(arr).view(np.uint8))
    return digest.hexdigest()


def stamp_hdf5_content_hash(path: str | Path, dataset_name: str = "data") -> "DatasetContract":
    """Compute the HDF5 dataset payload hash and update the embedded contract."""

    import h5py  # type: ignore

    digest = content_sha256_for_hdf5_dataset(path, dataset_name=dataset_name)
    contract = DatasetContract.from_hdf5(path, dataset_name=dataset_name, allow_legacy_dataset_contract=True)
    payload = contract.to_dict()
    payload["content_sha256"] = digest
    updated = DatasetContract.from_dict(
        payload,
        allow_legacy_dataset_contract=bool(payload.get("legacy_inferred")),
        allow_missing_source_gravity=bool(payload.get("legacy_inferred")),
        allow_legacy_derivative_convention=bool(payload.get("legacy_inferred")),
    )
    with h5py.File(path, "a") as handle:
        updated.write_hdf5_attrs(handle)
    return updated


def _repo_commit_sha() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parents[5]
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        value = completed.stdout.strip()
        return value or None
    except Exception:
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, default=_json_default)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attrs_to_dict(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _decode_attr(value) for key, value in attrs.items()}


def _get_first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return default


def _as_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _as_units(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, Mapping):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass
    return {}


def _columns(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    text = text.strip("[]")
    return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]


def _normalize_derivative(value: Any) -> Optional[str]:
    return None if value in (None, "") else str(value).strip()


@dataclass(frozen=True)
class DatasetContract:
    schema_version: int = DATASET_CONTRACT_SCHEMA_VERSION
    dataset_id: Optional[str] = None
    dataset_kind: str = "st_lrps_spatial_cloud"
    created_at_utc: Optional[str] = None
    generator_name: str = "spatial_cloud_generator"
    generator_version: Optional[str] = None
    repo_commit_sha: Optional[str] = None
    random_seed: Optional[int] = None
    n_samples: int = 0
    coordinate_frame: str = DEFAULT_COORDINATE_FRAME
    units: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UNITS))
    target_mode: str = "residual"
    baseline_kind: str = "spherical_harmonics"
    degree_min: Optional[int] = None
    degree_max: Optional[int] = None
    mu_si: float = MU_MOON_SI
    r_ref_m: float = R_MOON_SI
    a_sign: float = 1.0
    altitude_min_km: Optional[float] = None
    altitude_max_km: Optional[float] = None
    sampling_policy: dict[str, Any] = field(default_factory=dict)
    split_policy: dict[str, Any] = field(default_factory=dict)
    source_gravity_model: Optional[str] = None
    source_gravity_file_path: Optional[str] = None
    source_gravity_file_sha256: Optional[str] = None
    content_sha256: Optional[str] = None
    derivative_convention: Optional[str] = REQUIRED_DERIVATIVE_CONVENTION
    columns: list[str] = field(default_factory=lambda: ["x", "y", "z", "dU", "dax", "day", "daz"])
    dataset_layout: dict[str, Any] = field(default_factory=lambda: {"dataset_name": "data", "shape": None})
    legacy_inferred: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "dataset_kind", str(self.dataset_kind or "st_lrps_spatial_cloud"))
        object.__setattr__(self, "generator_name", str(self.generator_name or "spatial_cloud_generator"))
        object.__setattr__(self, "random_seed", None if self.random_seed is None else int(self.random_seed))
        object.__setattr__(self, "n_samples", int(self.n_samples or 0))
        object.__setattr__(self, "coordinate_frame", str(self.coordinate_frame or ""))
        object.__setattr__(self, "units", dict(self.units or {}))
        object.__setattr__(self, "target_mode", str(self.target_mode or "").strip().lower())
        object.__setattr__(self, "baseline_kind", str(self.baseline_kind or "").strip().lower())
        object.__setattr__(self, "degree_min", None if self.degree_min is None else int(self.degree_min))
        object.__setattr__(self, "degree_max", None if self.degree_max is None else int(self.degree_max))
        object.__setattr__(self, "mu_si", float(self.mu_si))
        object.__setattr__(self, "r_ref_m", float(self.r_ref_m))
        object.__setattr__(self, "a_sign", float(self.a_sign))
        object.__setattr__(self, "altitude_min_km", None if self.altitude_min_km is None else float(self.altitude_min_km))
        object.__setattr__(self, "altitude_max_km", None if self.altitude_max_km is None else float(self.altitude_max_km))
        object.__setattr__(self, "sampling_policy", dict(self.sampling_policy or {}))
        object.__setattr__(self, "split_policy", dict(self.split_policy or {}))
        object.__setattr__(self, "derivative_convention", _normalize_derivative(self.derivative_convention))
        object.__setattr__(self, "columns", _columns(self.columns))
        object.__setattr__(self, "dataset_layout", dict(self.dataset_layout or {}))
        object.__setattr__(self, "legacy_inferred", bool(self.legacy_inferred))
        self.validate(
            allow_legacy_dataset_contract=bool(self.legacy_inferred),
            allow_missing_source_gravity=bool(self.legacy_inferred),
            allow_legacy_derivative_convention=bool(self.legacy_inferred),
        )

    def validate(
        self,
        *,
        allow_legacy_dataset_contract: bool = False,
        allow_missing_source_gravity: bool = False,
        allow_legacy_derivative_convention: bool = False,
    ) -> None:
        errors: list[str] = []
        warnings: list[str] = []

        if self.schema_version != DATASET_CONTRACT_SCHEMA_VERSION:
            errors.append(f"schema_version must be {DATASET_CONTRACT_SCHEMA_VERSION}")
        if self.target_mode not in TARGET_MODES:
            errors.append("target_mode must be 'residual' or 'full'")
        if self.baseline_kind not in BASELINE_KINDS:
            errors.append(f"baseline_kind must be one of {sorted(BASELINE_KINDS)}")
        if self.degree_min is None or self.degree_max is None:
            errors.append("degree_min and degree_max are required")
        elif self.target_mode == "residual" and int(self.degree_max) <= int(self.degree_min):
            errors.append("residual datasets require degree_max > degree_min")
        elif self.target_mode == "full" and int(self.degree_max) < 0:
            errors.append("full-field datasets require degree_max >= 0")
        if not self.coordinate_frame:
            errors.append("coordinate_frame is required")
        for key, expected in DEFAULT_UNITS.items():
            if self.units.get(key) != expected:
                errors.append(f"units.{key} must be {expected!r}")
        if self.a_sign not in (-1.0, 1.0):
            errors.append("a_sign must be +1.0 or -1.0")
        if self.mu_si <= 0.0 or not np.isfinite(self.mu_si):
            errors.append("mu_si must be positive and finite")
        if self.r_ref_m <= 0.0 or not np.isfinite(self.r_ref_m):
            errors.append("r_ref_m must be positive and finite")
        if self.altitude_min_km is None or self.altitude_max_km is None:
            errors.append("altitude_min_km and altitude_max_km are required")
        elif float(self.altitude_max_km) <= float(self.altitude_min_km):
            errors.append("altitude_max_km must exceed altitude_min_km")
        if self.n_samples <= 0:
            errors.append("n_samples must be positive")
        if not self.columns:
            errors.append("columns are required")
        elif len(self.columns) != 7:
            errors.append("ST-LRPS datasets must declare exactly 7 columns")
        if self.derivative_convention != REQUIRED_DERIVATIVE_CONVENTION:
            msg = (
                f"derivative_convention={self.derivative_convention!r} is unsafe; "
                f"expected {REQUIRED_DERIVATIVE_CONVENTION!r}"
            )
            if allow_legacy_derivative_convention:
                warnings.append(msg)
            else:
                errors.append(msg)
        if self.target_mode == "residual" and self.baseline_kind == "none":
            errors.append("residual datasets require a non-none baseline_kind")
        if self.target_mode == "residual" and not (self.source_gravity_model or self.source_gravity_file_path):
            msg = "source gravity model information is required for residual datasets"
            if allow_missing_source_gravity or allow_legacy_dataset_contract:
                warnings.append(msg)
            else:
                errors.append(msg)
        if self.legacy_inferred and not allow_legacy_dataset_contract:
            errors.append("dataset contract was inferred from legacy attrs; pass allow_legacy_dataset_contract=True")

        if errors:
            raise DatasetContractError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        allow_legacy_dataset_contract: bool = False,
        allow_missing_source_gravity: bool = False,
        allow_legacy_derivative_convention: bool = False,
    ) -> "DatasetContract":
        data = dict(payload)
        if "derivative_convention" not in data and "derivative_convention_version" in data:
            data["derivative_convention"] = data.get("derivative_convention_version")
        if "altitude_min_km" not in data and "alt_min_km" in data:
            data["altitude_min_km"] = data.get("alt_min_km")
        if "altitude_max_km" not in data and "alt_max_km" in data:
            data["altitude_max_km"] = data.get("alt_max_km")
        if "random_seed" not in data and "seed" in data:
            data["random_seed"] = data.get("seed")
        obj = cls(**{k: v for k, v in data.items() if k in {f.name for f in dataclasses.fields(cls)}})
        obj.validate(
            allow_legacy_dataset_contract=allow_legacy_dataset_contract,
            allow_missing_source_gravity=allow_missing_source_gravity,
            allow_legacy_derivative_convention=allow_legacy_derivative_convention,
        )
        return obj

    @classmethod
    def from_hdf5_attrs(
        cls,
        attrs: Mapping[str, Any],
        *,
        n_samples: Optional[int] = None,
        dataset_name: str = "data",
        shape: Optional[tuple[int, ...]] = None,
        allow_legacy_dataset_contract: bool = False,
        allow_missing_dataset_contract: bool = False,
        allow_legacy_derivative_convention: bool = False,
    ) -> "DatasetContract":
        mapping = _attrs_to_dict(attrs)
        raw_contract = _get_first(mapping, DATASET_CONTRACT_ATTR, "contract_json", "dataset_contract")
        if isinstance(raw_contract, str) and raw_contract.strip():
            try:
                payload = json.loads(raw_contract)
                if isinstance(payload, Mapping):
                    if n_samples is not None and not payload.get("n_samples"):
                        payload = {**payload, "n_samples": int(n_samples)}
                    return cls.from_dict(
                        payload,
                        allow_legacy_dataset_contract=allow_legacy_dataset_contract,
                        allow_missing_source_gravity=allow_legacy_dataset_contract or allow_missing_dataset_contract,
                        allow_legacy_derivative_convention=allow_legacy_derivative_convention,
                    )
            except Exception as exc:
                if not allow_missing_dataset_contract:
                    raise DatasetContractError(f"could not parse dataset contract JSON: {exc}") from exc

        if not (allow_legacy_dataset_contract or allow_missing_dataset_contract):
            raise DatasetContractError(
                "dataset is missing dataset_contract_json; pass allow_legacy_dataset_contract=True "
                "only for old datasets"
            )

        degree_min = _as_optional_int(_get_first(mapping, "degree_min"))
        degree_max = _as_optional_int(_get_first(mapping, "degree_max", "requested_degree"))
        target_mode = str(_get_first(mapping, "target_mode", default="")).strip().lower()
        if not target_mode:
            target_mode = "residual" if degree_min is not None and degree_min >= 0 else "full"
        baseline_kind = str(_get_first(mapping, "baseline_kind", default="")).strip().lower()
        if not baseline_kind:
            baseline_kind = "spherical_harmonics" if target_mode == "residual" and (degree_min or -1) >= 0 else "none"
        seed = _as_optional_int(_get_first(mapping, "random_seed", "seed", "base_seed"))
        columns = _columns(_get_first(mapping, "columns", default="[x,y,z,dU,dax,day,daz]"))
        obj = cls(
            schema_version=DATASET_CONTRACT_SCHEMA_VERSION,
            dataset_id=_get_first(mapping, "dataset_id", "suite_id"),
            dataset_kind=str(_get_first(mapping, "dataset_kind", default="st_lrps_spatial_cloud")),
            created_at_utc=_get_first(mapping, "created_at_utc"),
            generator_name=str(_get_first(mapping, "generator_name", "created_by", default="spatial_cloud_generator")),
            generator_version=_get_first(mapping, "generator_version"),
            repo_commit_sha=_get_first(mapping, "repo_commit_sha"),
            random_seed=seed,
            n_samples=int(n_samples or _as_optional_int(_get_first(mapping, "n_samples")) or 0),
            coordinate_frame=str(_get_first(mapping, "coordinate_frame", "frame", default=DEFAULT_COORDINATE_FRAME)),
            units=_as_units(_get_first(mapping, "units", default=json.dumps(DEFAULT_UNITS))),
            target_mode=target_mode,
            baseline_kind=baseline_kind,
            degree_min=degree_min,
            degree_max=degree_max,
            mu_si=float(_as_optional_float(_get_first(mapping, "mu_si", "resolved_mu_si")) or MU_MOON_SI),
            r_ref_m=float(_as_optional_float(_get_first(mapping, "r_ref_m", "resolved_r_ref_m")) or R_MOON_SI),
            a_sign=float(_as_optional_float(_get_first(mapping, "a_sign", "resolved_a_sign")) or 1.0),
            altitude_min_km=_as_optional_float(_get_first(mapping, "altitude_min_km", "alt_min_km")),
            altitude_max_km=_as_optional_float(_get_first(mapping, "altitude_max_km", "alt_max_km")),
            sampling_policy={
                "name": _get_first(mapping, "sampling_strategy", "sampling_policy"),
                "surface_bias_ratio": _as_optional_float(_get_first(mapping, "surface_bias_ratio")),
            },
            split_policy={"role": _get_first(mapping, "dataset_role", "split")},
            source_gravity_model=_get_first(mapping, "source_gravity_model", "gravity_model_path", "gfc_path"),
            source_gravity_file_path=_get_first(mapping, "source_gravity_file_path", "gravity_model_path", "gfc_path"),
            source_gravity_file_sha256=_get_first(mapping, "source_gravity_file_sha256"),
            content_sha256=_get_first(mapping, "content_sha256", "dataset_sha256"),
            derivative_convention=_get_first(
                mapping,
                "derivative_convention",
                "derivative_convention_version",
            ),
            columns=columns,
            dataset_layout={"dataset_name": dataset_name, "shape": list(shape) if shape is not None else None},
            legacy_inferred=True,
        )
        obj.validate(
            allow_legacy_dataset_contract=True,
            allow_missing_source_gravity=True,
            allow_legacy_derivative_convention=allow_legacy_derivative_convention,
        )
        return obj

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        *,
        dataset_name: str = "data",
        allow_legacy_dataset_contract: bool = False,
        allow_missing_dataset_contract: bool = False,
        allow_legacy_derivative_convention: bool = False,
    ) -> "DatasetContract":
        import h5py  # type: ignore

        with h5py.File(path, "r") as handle:
            if METADATA_GROUP in handle and "contract_json" in handle[METADATA_GROUP]:
                raw = handle[METADATA_GROUP]["contract_json"][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                payload = json.loads(str(raw))
                return cls.from_dict(
                    payload,
                    allow_legacy_dataset_contract=allow_legacy_dataset_contract,
                    allow_missing_source_gravity=allow_legacy_dataset_contract,
                    allow_legacy_derivative_convention=allow_legacy_derivative_convention,
                )
            name = dataset_name if dataset_name in handle else next(
                key for key in handle.keys() if hasattr(handle[key], "shape")
            )
            shape = tuple(int(v) for v in handle[name].shape)
            return cls.from_hdf5_attrs(
                handle.attrs,
                n_samples=shape[0],
                dataset_name=name,
                shape=shape,
                allow_legacy_dataset_contract=allow_legacy_dataset_contract,
                allow_missing_dataset_contract=allow_missing_dataset_contract,
                allow_legacy_derivative_convention=allow_legacy_derivative_convention,
            )

    def write_hdf5_attrs(
        self,
        handle: Any,
        *,
        generation_config: Optional[Mapping[str, Any]] = None,
        quality_report: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload = self.to_dict()
        text = _json_text(payload)
        handle.attrs[DATASET_CONTRACT_ATTR] = text
        handle.attrs["schema_version"] = int(self.schema_version)
        handle.attrs["dataset_kind"] = self.dataset_kind
        handle.attrs["dataset_id"] = self.dataset_id or ""
        handle.attrs["target_mode"] = self.target_mode
        handle.attrs["baseline_kind"] = self.baseline_kind
        handle.attrs["n_samples"] = int(self.n_samples)
        handle.attrs["degree_min"] = "" if self.degree_min is None else int(self.degree_min)
        handle.attrs["degree_max"] = "" if self.degree_max is None else int(self.degree_max)
        handle.attrs["mu_si"] = float(self.mu_si)
        handle.attrs["r_ref_m"] = float(self.r_ref_m)
        handle.attrs["a_sign_convention"] = "+1" if self.a_sign > 0 else "-1"
        handle.attrs["alt_min_km"] = "" if self.altitude_min_km is None else float(self.altitude_min_km)
        handle.attrs["alt_max_km"] = "" if self.altitude_max_km is None else float(self.altitude_max_km)
        handle.attrs["coordinate_frame"] = self.coordinate_frame
        handle.attrs["units"] = json.dumps(self.units, sort_keys=True)
        handle.attrs["derivative_convention_version"] = self.derivative_convention or ""
        handle.attrs["columns"] = "[" + ",".join(self.columns) + "]"
        handle.attrs["source_gravity_model"] = self.source_gravity_model or ""
        handle.attrs["source_gravity_file_path"] = self.source_gravity_file_path or ""
        handle.attrs["source_gravity_file_sha256"] = self.source_gravity_file_sha256 or ""
        handle.attrs["content_sha256"] = self.content_sha256 or ""
        meta = handle.require_group(METADATA_GROUP)
        _write_scalar_text_dataset(meta, "contract_json", text)
        if generation_config is not None:
            _write_scalar_text_dataset(meta, "generation_json", _json_text(dict(generation_config)))
        if quality_report is not None:
            _write_scalar_text_dataset(meta, "quality_report_json", _json_text(dict(quality_report)))

    def compatibility_report(self, other: "DatasetContract | Mapping[str, Any]") -> dict[str, Any]:
        rhs = other if isinstance(other, DatasetContract) else DatasetContract.from_dict(other)
        errors: list[str] = []
        warnings: list[str] = []
        for key in ("target_mode", "baseline_kind", "degree_min", "degree_max", "coordinate_frame"):
            if getattr(self, key) != getattr(rhs, key):
                errors.append(f"{key} mismatch: {getattr(self, key)!r} != {getattr(rhs, key)!r}")
        for key in ("mu_si", "r_ref_m", "a_sign"):
            if abs(float(getattr(self, key)) - float(getattr(rhs, key))) > (1.0 if key != "a_sign" else 0.0):
                errors.append(f"{key} mismatch: {getattr(self, key)!r} != {getattr(rhs, key)!r}")
        if self.units != rhs.units:
            errors.append("units mismatch")
        if self.derivative_convention != rhs.derivative_convention:
            errors.append("derivative_convention mismatch")
        if self.content_sha256 and rhs.content_sha256 and self.content_sha256 != rhs.content_sha256:
            errors.append("content_sha256 mismatch")
        if not self.source_gravity_file_sha256:
            warnings.append("source_gravity_file_sha256 missing")
        return {
            "compatible": not errors,
            "errors": errors,
            "warnings": warnings,
            "left": self.to_dict(),
            "right": rhs.to_dict(),
        }

    def require_compatible(self, other: "DatasetContract | Mapping[str, Any]", *, strict: bool = True) -> dict[str, Any]:
        report = self.compatibility_report(other)
        if strict and report["errors"]:
            raise DatasetContractError("; ".join(report["errors"]))
        return report


def _write_scalar_text_dataset(group: Any, name: str, text: str) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=np.bytes_(text))


def contract_from_generation_attrs(attrs: Mapping[str, Any], *, n_samples: int, dataset_name: str = "data") -> DatasetContract:
    return DatasetContract.from_hdf5_attrs(
        attrs,
        n_samples=n_samples,
        dataset_name=dataset_name,
        shape=(int(n_samples), 7),
        allow_legacy_dataset_contract=True,
        allow_missing_dataset_contract=True,
    )


def build_contract_payload_for_generator(
    *,
    dataset_id: Optional[str],
    n_samples: int,
    degree_min: int,
    degree_max: int,
    target_mode: str,
    baseline_kind: str,
    mu_si: float,
    r_ref_m: float,
    altitude_min_km: float,
    altitude_max_km: float,
    random_seed: int,
    sampling_policy: Mapping[str, Any],
    source_gravity_model: str | None,
    source_gravity_file_path: str | None,
    source_gravity_file_sha256: str | None,
    generator_version: str,
    columns: list[str],
) -> dict[str, Any]:
    return DatasetContract(
        dataset_id=dataset_id,
        created_at_utc=utc_now_iso(),
        generator_name="spatial_cloud_generator",
        generator_version=generator_version,
        repo_commit_sha=_repo_commit_sha(),
        random_seed=random_seed,
        n_samples=n_samples,
        target_mode=target_mode,
        baseline_kind=baseline_kind,
        degree_min=degree_min,
        degree_max=degree_max,
        mu_si=mu_si,
        r_ref_m=r_ref_m,
        altitude_min_km=altitude_min_km,
        altitude_max_km=altitude_max_km,
        sampling_policy=dict(sampling_policy),
        source_gravity_model=source_gravity_model,
        source_gravity_file_path=source_gravity_file_path,
        source_gravity_file_sha256=source_gravity_file_sha256,
        columns=columns,
        dataset_layout={"dataset_name": "data", "shape": [int(n_samples), 7]},
    ).to_dict()


def ensure_output_path_allowed(path: str | Path, *, overwrite: bool = False) -> Path:
    out = Path(path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[5]
    src_root = repo_root / "src"
    try:
        out.relative_to(src_root)
    except ValueError:
        pass
    else:
        raise ValueError(f"Refusing to write generated dataset inside source package directory: {out}")
    if out.exists() and not overwrite:
        raise FileExistsError(f"Dataset output already exists: {out}. Pass --overwrite to replace it.")
    return out


__all__ = [
    "BASELINE_KINDS",
    "DATASET_CONTRACT_ATTR",
    "DATASET_CONTRACT_SCHEMA_VERSION",
    "DEFAULT_COORDINATE_FRAME",
    "DEFAULT_UNITS",
    "DatasetContract",
    "DatasetContractError",
    "REQUIRED_DERIVATIVE_CONVENTION",
    "TARGET_MODES",
    "build_contract_payload_for_generator",
    "content_sha256_for_hdf5_dataset",
    "contract_from_generation_attrs",
    "ensure_output_path_allowed",
    "sha256_file",
    "stamp_hdf5_content_hash",
    "utc_now_iso",
]
