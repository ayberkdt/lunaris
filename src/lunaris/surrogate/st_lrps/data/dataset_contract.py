"""Versioned ST-LRPS dataset contract utilities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.common.provenance import sha256_file as _sha256_file
from lunaris.common.provenance import utc_now_iso
from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI

# Canonical target/frame/derivative-convention names live in
# lunaris.surrogate.st_lrps.shared.contracts; re-exported here for dataset code.
from lunaris.surrogate.st_lrps.shared.contracts import (
    BASELINE_KINDS,
    MOON_FIXED_FRAME,
    REQUIRED_DERIVATIVE_CONVENTION,
    TARGET_MODES,
)

DATASET_CONTRACT_SCHEMA_VERSION = 1
# Spherical-harmonic phase/normalization convention used to generate the gravity
# labels. Must match the runtime engine (lunaris.physics.spherical_harmonics):
# 4pi geodesy normalization with NO Condon-Shortley phase. Datasets generated
# before this was enforced (the generator used a negative sectoral recurrence =
# (-1)^m phase) have sign-flipped odd-order labels and MUST be regenerated.
REQUIRED_SH_PHASE_CONVENTION = "4pi_geodesy_no_condon_shortley_v1"
GRAVITY_LABEL_ENGINE_VERSION = "lunaris_sh_v2"
DEFAULT_COORDINATE_FRAME = MOON_FIXED_FRAME
DEFAULT_UNITS = {"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"}
DATASET_CONTRACT_ATTR = "dataset_contract_json"
METADATA_GROUP = "metadata"


class DatasetContractError(ValueError):
    """Raised when dataset metadata is missing, ambiguous, or unsafe."""


def sha256_file(path: str | Path | None) -> str | None:
    return _sha256_file(path, missing_ok=True)


def content_sha256_for_hdf5_dataset(path: str | Path, dataset_name: str = "data", *, chunk_rows: int = 65536) -> str:
    import h5py

    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        ds = handle[dataset_name]
        for start in range(0, int(ds.shape[0]), int(chunk_rows)):
            arr = np.asarray(ds[start : start + int(chunk_rows)])
            digest.update(np.ascontiguousarray(arr).view(np.uint8).tobytes())
    return digest.hexdigest()


def stamp_hdf5_content_hash(path: str | Path, dataset_name: str = "data") -> DatasetContract:
    """Compute the HDF5 dataset payload hash and update the embedded contract."""

    import h5py

    digest = content_sha256_for_hdf5_dataset(path, dataset_name=dataset_name)
    contract = DatasetContract.from_hdf5(path, dataset_name=dataset_name)
    payload = contract.to_dict()
    payload["content_sha256"] = digest
    updated = DatasetContract.from_dict(payload)
    with h5py.File(path, "a") as handle:
        updated.write_hdf5_attrs(handle)
    return updated


def _repo_commit_sha() -> str | None:
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


def _columns(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    text = text.strip("[]")
    return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]


def _normalize_derivative(value: Any) -> str | None:
    return None if value in (None, "") else str(value).strip()


@dataclass(frozen=True)
class DatasetContract:
    schema_version: int = DATASET_CONTRACT_SCHEMA_VERSION
    dataset_id: str | None = None
    dataset_kind: str = "st_lrps_spatial_cloud"
    created_at_utc: str | None = None
    generator_name: str = "spatial_cloud_generator"
    generator_version: str | None = None
    repo_commit_sha: str | None = None
    random_seed: int | None = None
    n_samples: int = 0
    coordinate_frame: str = DEFAULT_COORDINATE_FRAME
    units: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UNITS))
    target_mode: str = "residual"
    baseline_kind: str = "spherical_harmonics"
    degree_min: int | None = None
    degree_max: int | None = None
    mu_si: float = MU_MOON_SI
    r_ref_m: float = R_MOON_SI
    a_sign: float = 1.0
    altitude_min_km: float | None = None
    altitude_max_km: float | None = None
    sampling_policy: dict[str, Any] = field(default_factory=dict)
    split_policy: dict[str, Any] = field(default_factory=dict)
    source_gravity_model: str | None = None
    source_gravity_file_path: str | None = None
    source_gravity_file_sha256: str | None = None
    content_sha256: str | None = None
    derivative_convention: str | None = REQUIRED_DERIVATIVE_CONVENTION
    # Default None (not the required value) so datasets predating phase enforcement
    # fail closed: only freshly generated datasets carry the stamped convention.
    spherical_harmonic_convention: str | None = None
    gravity_label_engine_version: str | None = None
    columns: list[str] = field(default_factory=lambda: ["x", "y", "z", "dU", "dax", "day", "daz"])
    dataset_layout: dict[str, Any] = field(default_factory=lambda: {"dataset_name": "data", "shape": None})
    _skip_initial_validation: InitVar[bool] = False

    def __post_init__(self, _skip_initial_validation: bool) -> None:
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
        if not _skip_initial_validation:
            self.validate()

    def validate(self) -> None:
        errors: list[str] = []

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
            errors.append(msg)
        if self.spherical_harmonic_convention != REQUIRED_SH_PHASE_CONVENTION:
            msg = (
                f"spherical_harmonic_convention={self.spherical_harmonic_convention!r} is unsafe; "
                f"expected {REQUIRED_SH_PHASE_CONVENTION!r}. Datasets generated before the "
                "Condon-Shortley phase fix have sign-flipped odd-order gravity labels and must "
                "be regenerated."
            )
            errors.append(msg)
        if self.gravity_label_engine_version != GRAVITY_LABEL_ENGINE_VERSION:
            msg = (
                f"gravity_label_engine_version={self.gravity_label_engine_version!r} is unsafe; "
                f"expected {GRAVITY_LABEL_ENGINE_VERSION!r}"
            )
            errors.append(msg)
        if self.target_mode == "residual" and self.baseline_kind == "none":
            errors.append("residual datasets require a non-none baseline_kind")
        if self.target_mode == "residual" and not (self.source_gravity_model or self.source_gravity_file_path):
            msg = "source gravity model information is required for residual datasets"
            errors.append(msg)
        if errors:
            raise DatasetContractError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetContract:
        data = dict(payload)
        if "derivative_convention" not in data and "derivative_convention_version" in data:
            data["derivative_convention"] = data.get("derivative_convention_version")
        if "altitude_min_km" not in data and "alt_min_km" in data:
            data["altitude_min_km"] = data.get("alt_min_km")
        if "altitude_max_km" not in data and "alt_max_km" in data:
            data["altitude_max_km"] = data.get("alt_max_km")
        if "random_seed" not in data and "seed" in data:
            data["random_seed"] = data.get("seed")
        obj = cls(
            **{k: v for k, v in data.items() if k in {f.name for f in dataclasses.fields(cls)}},
            _skip_initial_validation=True,
        )
        obj.validate()
        return obj

    @classmethod
    def from_hdf5_attrs(
        cls,
        attrs: Mapping[str, Any],
        *,
        n_samples: int | None = None,
        dataset_name: str = "data",
        shape: tuple[int, ...] | None = None,
    ) -> DatasetContract:
        mapping = _attrs_to_dict(attrs)
        raw_contract = _get_first(mapping, DATASET_CONTRACT_ATTR, "contract_json", "dataset_contract")
        if isinstance(raw_contract, str) and raw_contract.strip():
            try:
                payload = json.loads(raw_contract)
                if isinstance(payload, Mapping):
                    if n_samples is not None and not payload.get("n_samples"):
                        payload = {**payload, "n_samples": int(n_samples)}
                    return cls.from_dict(payload)
            except Exception as exc:
                raise DatasetContractError(f"could not parse dataset contract JSON: {exc}") from exc

        raise DatasetContractError(
            "dataset is missing dataset_contract_json. Legacy metadata inference has been removed; "
            "regenerate the dataset with the current generator."
        )

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        *,
        dataset_name: str = "data",
    ) -> DatasetContract:
        import h5py

        with h5py.File(path, "r") as handle:
            if METADATA_GROUP in handle and "contract_json" in handle[METADATA_GROUP]:
                raw = handle[METADATA_GROUP]["contract_json"][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                payload = json.loads(str(raw))
                return cls.from_dict(payload)
            name = dataset_name if dataset_name in handle else next(
                key for key in handle.keys() if hasattr(handle[key], "shape")
            )
            shape = tuple(int(v) for v in handle[name].shape)
            return cls.from_hdf5_attrs(
                handle.attrs,
                n_samples=shape[0],
                dataset_name=name,
                shape=shape,
            )

    def write_hdf5_attrs(
        self,
        handle: Any,
        *,
        generation_config: Mapping[str, Any] | None = None,
        quality_report: Mapping[str, Any] | None = None,
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
        handle.attrs["spherical_harmonic_convention"] = self.spherical_harmonic_convention or ""
        handle.attrs["gravity_label_engine_version"] = self.gravity_label_engine_version or ""
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

    def compatibility_report(self, other: DatasetContract | Mapping[str, Any]) -> dict[str, Any]:
        rhs = other if isinstance(other, DatasetContract) else DatasetContract.from_dict(other)
        errors: list[str] = []
        warnings: list[str] = []
        errors.extend(
            f"{key} mismatch: {getattr(self, key)!r} != {getattr(rhs, key)!r}"
            for key in ("target_mode", "baseline_kind", "degree_min", "degree_max", "coordinate_frame")
            if getattr(self, key) != getattr(rhs, key)
        )
        errors.extend(
            f"{key} mismatch: {getattr(self, key)!r} != {getattr(rhs, key)!r}"
            for key in ("mu_si", "r_ref_m", "a_sign")
            if abs(float(getattr(self, key)) - float(getattr(rhs, key))) > (1.0 if key != "a_sign" else 0.0)
        )
        if self.units != rhs.units:
            errors.append("units mismatch")
        if self.derivative_convention != rhs.derivative_convention:
            errors.append("derivative_convention mismatch")
        errors.extend(
            f"{key} mismatch: {getattr(self, key)!r} != {getattr(rhs, key)!r}"
            for key in ("spherical_harmonic_convention", "gravity_label_engine_version")
            if getattr(self, key) != getattr(rhs, key)
        )
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

    def require_compatible(self, other: DatasetContract | Mapping[str, Any], *, strict: bool = True) -> dict[str, Any]:
        report = self.compatibility_report(other)
        if strict and report["errors"]:
            raise DatasetContractError("; ".join(report["errors"]))
        return report


def _write_scalar_text_dataset(group: Any, name: str, text: str) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=np.bytes_(text))


def build_contract_payload_for_generator(
    *,
    dataset_id: str | None,
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
        spherical_harmonic_convention=REQUIRED_SH_PHASE_CONVENTION,
        gravity_label_engine_version=GRAVITY_LABEL_ENGINE_VERSION,
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
    "GRAVITY_LABEL_ENGINE_VERSION",
    "REQUIRED_DERIVATIVE_CONVENTION",
    "REQUIRED_SH_PHASE_CONVENTION",
    "TARGET_MODES",
    "build_contract_payload_for_generator",
    "content_sha256_for_hdf5_dataset",
    "ensure_output_path_allowed",
    "sha256_file",
    "stamp_hdf5_content_hash",
    "utc_now_iso",
]
