"""Explicit target and artifact contracts for ST-LRPS lunar surrogate artifacts.

The target contract separates the harmonic degree range of the reference/high
degree fields from whether the dataset stores residual labels or full-field
labels. Artifact contracts are generated from fully resolved training configs
and embedded in every current checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from lunaris.common.lunar_data import (
    MU_MOON_SI,
    R_MOON_SI,
    is_lunar_body_signature,
)

REQUIRED_DERIVATIVE_CONVENTION = "dP_dphi_corrected_v1"
# Canonical body-fixed Cartesian frame name for every ST-LRPS dataset/artifact.
MOON_FIXED_FRAME = "moon_fixed_cartesian"
LUNAR_BODY_ALIASES = frozenset({"moon", "lunar", "selene"})
TARGET_MODES = frozenset({"residual", "full"})
BASELINE_KINDS = frozenset({"none", "point_mass", "spherical_harmonics"})
ARTIFACT_CONTRACT_SCHEMA_VERSION = 1
RUNTIME_MODEL_KINDS = frozenset({"potential_autograd"})
# Model kinds that once existed on main but are no longer loadable. Kept as an
# explicit set so old artifacts are rejected with a pointer to the archive
# instead of a generic "unknown kind" error.
ARCHIVED_RUNTIME_MODEL_KINDS = frozenset({"force_direct"})
PREDICTION_KINDS = frozenset({"potential", "residual_potential", "total"})


class ArtifactContractError(RuntimeError):
    """Raised when an ST-LRPS artifact contract is invalid or incompatible."""


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _clean_str(value: Any, default: str) -> str:
    text = str(value if value is not None else default).strip().lower()
    return text or str(default)


@dataclass(frozen=True)
class TargetContract:
    """First-class target semantics for ST-LRPS training/evaluation/runtime."""

    central_body: str
    target_mode: str
    base_degree: int
    target_degree: int
    baseline_kind: str
    unit_system: str
    frame: str
    derivative_convention_version: str
    a_sign: float
    mu_si: float
    r_ref_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "central_body", _clean_str(self.central_body, "moon"))
        object.__setattr__(self, "target_mode", _clean_str(self.target_mode, "residual"))
        object.__setattr__(self, "baseline_kind", _clean_str(self.baseline_kind, "none"))
        object.__setattr__(self, "unit_system", _clean_str(self.unit_system, "si"))
        object.__setattr__(self, "frame", str(self.frame or MOON_FIXED_FRAME).strip())
        object.__setattr__(
            self,
            "derivative_convention_version",
            str(self.derivative_convention_version or REQUIRED_DERIVATIVE_CONVENTION).strip(),
        )
        object.__setattr__(self, "base_degree", int(self.base_degree))
        object.__setattr__(self, "target_degree", int(self.target_degree))
        object.__setattr__(self, "a_sign", float(self.a_sign))
        object.__setattr__(self, "mu_si", float(self.mu_si))
        object.__setattr__(self, "r_ref_m", float(self.r_ref_m))
        self.validate()

    def validate(self) -> None:
        if self.target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be 'residual' or 'full', got {self.target_mode!r}.")
        if self.central_body not in LUNAR_BODY_ALIASES:
            raise ValueError(
                f"central_body={self.central_body!r} is not lunar-compatible; "
                "expected one of 'moon', 'lunar', or 'selene'."
            )
        if self.baseline_kind not in BASELINE_KINDS:
            raise ValueError(
                f"baseline_kind must be one of {sorted(BASELINE_KINDS)}, got {self.baseline_kind!r}."
            )
        if self.target_mode == "residual":
            if self.target_degree <= self.base_degree:
                raise ValueError(
                    f"Residual targets require target_degree > base_degree; got "
                    f"{self.target_degree} <= {self.base_degree}."
                )
            if self.base_degree < 0:
                raise ValueError("Residual SH contracts require base_degree >= 0.")
        if self.a_sign not in (-1.0, 1.0):
            raise ValueError(f"a_sign must be +1.0 or -1.0, got {self.a_sign!r}.")
        if not is_lunar_body_signature(mu_si=self.mu_si, r_ref_m=self.r_ref_m):
            raise ValueError(
                "TargetContract body constants do not look lunar: "
                f"mu_si={self.mu_si!r}, r_ref_m={self.r_ref_m!r}."
            )
        if self.derivative_convention_version != REQUIRED_DERIVATIVE_CONVENTION:
            raise ValueError(
                "TargetContract derivative_convention_version must be "
                f"{REQUIRED_DERIVATIVE_CONVENTION!r}; got "
                f"{self.derivative_convention_version!r}."
            )

    @property
    def is_residual(self) -> bool:
        return self.target_mode == "residual"

    @property
    def requires_baseline(self) -> bool:
        return self.baseline_kind != "none"

    @property
    def baseline_description(self) -> str:
        if self.baseline_kind == "none":
            return "no analytical baseline"
        if self.baseline_kind == "point_mass":
            return "point-mass baseline"
        return f"spherical-harmonics baseline through degree {self.base_degree}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetContract:
        return cls(
            central_body=payload.get("central_body", "moon"),
            target_mode=payload.get("target_mode", "residual"),
            base_degree=_as_int(payload.get("base_degree"), -1),
            target_degree=_as_int(payload.get("target_degree"), -1),
            baseline_kind=payload.get("baseline_kind", "none"),
            unit_system=payload.get("unit_system", "si"),
            frame=payload.get("frame", MOON_FIXED_FRAME),
            derivative_convention_version=payload.get(
                "derivative_convention_version",
                REQUIRED_DERIVATIVE_CONVENTION,
            ),
            a_sign=_as_float(payload.get("a_sign"), 1.0),
            mu_si=_as_float(payload.get("mu_si"), MU_MOON_SI),
            r_ref_m=_as_float(payload.get("r_ref_m"), R_MOON_SI),
        )

    @classmethod
    def from_dataset_meta(
        cls,
        meta: Any,
        resolved_mu_si: float,
        resolved_r_ref_m: float,
        a_sign: float,
    ) -> TargetContract:
        target_mode = getattr(meta, "target_mode", None)
        base_degree = _as_int(getattr(meta, "degree_min", None), -1)
        if not target_mode:
            raise ValueError(
                "Dataset metadata is missing target_mode. Regenerate the dataset "
                "with the current contract schema."
            )
        target_mode = _clean_str(target_mode, "residual")
        target_degree = _as_int(
            getattr(meta, "degree_max", None),
            _as_int(getattr(meta, "requested_degree", None), -1),
        )
        baseline_kind = _baseline_kind_for(target_mode, base_degree)
        deriv = getattr(meta, "derivative_convention_version", None)
        return cls(
            central_body=getattr(meta, "central_body", None) or "moon",
            target_mode=target_mode,
            base_degree=base_degree,
            target_degree=target_degree,
            baseline_kind=baseline_kind,
            unit_system=getattr(meta, "unit_system", None) or "unknown",
            frame=MOON_FIXED_FRAME,
            derivative_convention_version=(deriv or REQUIRED_DERIVATIVE_CONVENTION),
            a_sign=float(a_sign),
            mu_si=float(resolved_mu_si),
            r_ref_m=float(resolved_r_ref_m),
        )

    @classmethod
    def from_resolved_config(
        cls,
        config: Mapping[str, Any],
        *,
        resolved_mu_si: float | None = None,
        resolved_r_ref_m: float | None = None,
        a_sign: float | None = None,
    ) -> TargetContract:
        """Build a target contract from a fully resolved training config."""

        if isinstance(config.get("target_contract"), Mapping):
            return cls.from_dict(config["target_contract"])

        _raw_meta = config.get("dataset_meta")
        dataset_meta: Mapping[str, Any] = _raw_meta if isinstance(_raw_meta, Mapping) else {}
        base_degree = _as_int(config.get("degree_min", dataset_meta.get("degree_min")), -1)
        target_degree = _as_int(
            config.get("degree_max", dataset_meta.get("degree_max", dataset_meta.get("requested_degree"))),
            max(base_degree + 1, 0),
        )
        target_mode = _clean_str(
            config.get("target_mode", dataset_meta.get("target_mode")),
            "residual" if base_degree >= 0 else "full",
        )
        return cls(
            central_body=config.get("central_body", dataset_meta.get("central_body", "moon")),
            target_mode=target_mode,
            base_degree=base_degree,
            target_degree=target_degree,
            baseline_kind=_baseline_kind_for(target_mode, base_degree),
            unit_system=config.get("unit_system", dataset_meta.get("unit_system", "unknown")),
            frame=config.get("frame", MOON_FIXED_FRAME),
            derivative_convention_version=config.get(
                "derivative_convention_version",
                dataset_meta.get("derivative_convention_version", REQUIRED_DERIVATIVE_CONVENTION),
            ),
            a_sign=float(a_sign if a_sign is not None else config.get("resolved_a_sign", config.get("a_sign", 1.0))),
            mu_si=float(
                resolved_mu_si
                if resolved_mu_si is not None
                else config.get("resolved_mu_si", config.get("mu_si", dataset_meta.get("mu_si", MU_MOON_SI)))
            ),
            r_ref_m=float(
                resolved_r_ref_m
                if resolved_r_ref_m is not None
                else config.get("resolved_r_ref_m", config.get("r_ref_m", dataset_meta.get("r_ref_m", R_MOON_SI)))
            ),
        )


def _baseline_kind_for(target_mode: str, base_degree: int) -> str:
    mode = _clean_str(target_mode, "residual")
    if mode == "residual":
        return "spherical_harmonics"
    if int(base_degree) < 0:
        return "point_mass"
    if int(base_degree) == 0:
        return "point_mass"
    return "spherical_harmonics"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


def _json_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _input_encoding_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "model_preset",
        "input_feature_dim",
        "embedding_type",
        "use_fourier",
        "fourier_append_raw",
        "fourier_n_features",
        "fourier_sigma",
        "fourier_seed",
        "use_sh_encoding",
        "sh_encoding_degree",
        "sh_append_raw",
        "use_radial_separation",
        "radial_append_raw",
        "use_radial_decay_encoding",
        "radial_decay_max_power",
        "radial_decay_append_raw",
        "use_physical_radial_decay_encoding",
        "physical_radial_decay_max_power",
        "physical_radial_decay_append_raw",
        "physical_radial_decay_include_unit",
        "physical_radial_decay_include_r_scaled",
        "use_real_sh_basis",
        "real_sh_degree",
        "real_sh_append_raw",
        "real_sh_include_radial",
    )
    return {key: config.get(key) for key in keys if key in config}


def _scaler_contract_from_payload(value: Any) -> dict[str, Any]:
    payload = _mapping(value)
    if not payload:
        return {}
    provenance = _mapping(payload.get("provenance"))
    return {
        "schema_version": 1,
        "kind": payload.get("kind", "isometric"),
        "x": _mapping(payload.get("x")),
        "u": _mapping(payload.get("u")),
        "a": _mapping(payload.get("a")),
        "provenance": provenance,
        "sha256": _json_hash(payload) if payload else None,
    }


#: Dataset digest keys across schema generations, canonical name first. The
#: canonical ``DatasetContract`` records the HDF5 content digest as
#: ``content_sha256``; older manifests/configs used ``dataset_sha256``,
#: ``sha256`` or ``dataset_hash``.
_DATASET_HASH_KEYS: tuple[str, ...] = ("content_sha256", "dataset_sha256", "sha256", "dataset_hash")


def resolve_dataset_hash(*sources: Any) -> str | None:
    """Resolve the dataset digest from mappings that may use legacy key names.

    Sources are consulted in order; within each, the canonical
    ``content_sha256`` wins over the legacy names. Readers must go through this
    helper so a schema rename never reads as a missing hash (a real absent
    digest and a key-name mismatch would otherwise be indistinguishable).
    """
    for source in sources:
        mapping = _mapping(source)
        if not mapping:
            continue
        for key in _DATASET_HASH_KEYS:
            value = mapping.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return None


def _dataset_contract_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    meta = _mapping(config.get("dataset_meta") or config.get("dataset_contract") or config.get("dataset"))
    if not meta:
        return {}
    return {
        "schema_version": int(meta.get("schema_version", 1) or 1),
        "dataset_kind": meta.get("dataset_kind", "st_lrps_spatial_cloud"),
        "dataset_name": meta.get("dataset_name") or config.get("dataset_name"),
        "dataset_sha256": resolve_dataset_hash(meta, config),
        "target_mode": meta.get("target_mode") or config.get("target_mode"),
        "baseline_kind": meta.get("baseline_kind") or config.get("baseline_kind"),
        "degree_min": meta.get("degree_min") if meta.get("degree_min") is not None else config.get("degree_min"),
        "degree_max": meta.get("degree_max") if meta.get("degree_max") is not None else config.get("degree_max"),
        "mu_si": meta.get("mu_si") or config.get("resolved_mu_si") or config.get("mu_si"),
        "r_ref_m": meta.get("r_ref_m") or config.get("resolved_r_ref_m") or config.get("r_ref_m"),
        "a_sign": meta.get("a_sign") or config.get("resolved_a_sign") or config.get("a_sign"),
        "altitude_min_km": meta.get("alt_min_km") or meta.get("altitude_min_km"),
        "altitude_max_km": meta.get("alt_max_km") or meta.get("altitude_max_km"),
        "coordinate_frame": meta.get("coordinate_frame") or meta.get("frame") or MOON_FIXED_FRAME,
        "units": meta.get("units") or {"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"},
        "generator_version": meta.get("generator_version") or meta.get("created_by"),
        "source_gravity_model": meta.get("source_gravity_model") or meta.get("gravity_model_path"),
        "source_gravity_file_sha256": meta.get("source_gravity_file_sha256"),
        "n_samples": meta.get("n_samples"),
        "split": meta.get("split"),
        "derivative_convention_version": meta.get("derivative_convention_version"),
    }


@dataclass(frozen=True)
class ArtifactContract:
    """Versioned scientific contract for ST-LRPS artifacts.

    This is intentionally broader than :class:`TargetContract`: it binds target
    semantics to scaler, dataset, input encoding, and architecture identity so
    runtime/evaluation/benchmark code can reject invalid combinations.
    """

    schema_version: int
    target_mode: str
    baseline_kind: str
    base_degree: int
    target_degree: int
    runtime_model_kind: str
    prediction_kind: str
    mu_si: float
    r_ref_m: float
    a_sign: float
    altitude_min_km: float | None
    altitude_max_km: float | None
    input_encoding: dict[str, Any]
    scaler_contract: dict[str, Any]
    dataset_contract: dict[str, Any]
    output_dim: int = 1
    architecture_signature: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "target_mode", _clean_str(self.target_mode, ""))
        object.__setattr__(self, "baseline_kind", _clean_str(self.baseline_kind, "none"))
        object.__setattr__(self, "base_degree", int(self.base_degree))
        object.__setattr__(self, "target_degree", int(self.target_degree))
        object.__setattr__(self, "runtime_model_kind", _clean_str(self.runtime_model_kind, ""))
        object.__setattr__(self, "prediction_kind", _clean_str(self.prediction_kind, "residual_potential"))
        object.__setattr__(self, "mu_si", float(self.mu_si))
        object.__setattr__(self, "r_ref_m", float(self.r_ref_m))
        object.__setattr__(self, "a_sign", float(self.a_sign))
        object.__setattr__(
            self,
            "altitude_min_km",
            None if self.altitude_min_km is None else float(self.altitude_min_km),
        )
        object.__setattr__(
            self,
            "altitude_max_km",
            None if self.altitude_max_km is None else float(self.altitude_max_km),
        )
        object.__setattr__(self, "input_encoding", _mapping(self.input_encoding))
        object.__setattr__(self, "scaler_contract", _mapping(self.scaler_contract))
        object.__setattr__(self, "dataset_contract", _mapping(self.dataset_contract))
        object.__setattr__(self, "output_dim", int(self.output_dim))
        object.__setattr__(
            self,
            "architecture_signature",
            None if self.architecture_signature in (None, "") else str(self.architecture_signature),
        )
        self.validate()

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != ARTIFACT_CONTRACT_SCHEMA_VERSION:
            errors.append(
                f"schema_version must be {ARTIFACT_CONTRACT_SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        if self.target_mode not in TARGET_MODES:
            errors.append("target_mode must be 'residual' or 'full'")
        if self.baseline_kind not in BASELINE_KINDS:
            errors.append(f"baseline_kind must be one of {sorted(BASELINE_KINDS)}")
        if self.runtime_model_kind in ARCHIVED_RUNTIME_MODEL_KINDS:
            errors.append(
                f"runtime_model_kind={self.runtime_model_kind!r} is archived in the "
                "experimental/force-direct-archive branch and cannot be loaded on main; "
                "only 'potential_autograd' artifacts are supported"
            )
        elif self.runtime_model_kind not in RUNTIME_MODEL_KINDS:
            errors.append(
                f"unsupported runtime_model_kind={self.runtime_model_kind!r}; "
                "expected 'potential_autograd'"
            )
        if self.prediction_kind not in PREDICTION_KINDS:
            errors.append(f"unsupported prediction_kind={self.prediction_kind!r}")
        if self.output_dim <= 0:
            errors.append("output_dim must be positive")
        if self.runtime_model_kind == "potential_autograd":
            if self.output_dim != 1:
                errors.append("potential_autograd artifacts must have output_dim=1")
            if self.prediction_kind not in {"potential", "residual_potential", "total"}:
                errors.append("potential_autograd artifacts must predict scalar potential targets")
        if self.target_mode == "residual":
            if self.baseline_kind == "none":
                errors.append("residual contracts require a non-none baseline_kind")
            if self.base_degree < 0:
                errors.append("residual contracts require base_degree >= 0")
            if self.target_degree <= self.base_degree:
                errors.append("residual contracts require target_degree > base_degree")
        elif self.target_degree < 0:
            errors.append("full-field contracts require target_degree >= 0")
        if self.a_sign not in (-1.0, 1.0):
            errors.append("a_sign must be +1.0 or -1.0")
        if not is_lunar_body_signature(mu_si=self.mu_si, r_ref_m=self.r_ref_m):
            errors.append(
                f"mu_si/r_ref_m do not look lunar: mu_si={self.mu_si!r}, r_ref_m={self.r_ref_m!r}"
            )
        if self.altitude_min_km is not None and self.altitude_max_km is not None:
            if self.altitude_max_km <= self.altitude_min_km:
                errors.append("altitude_max_km must exceed altitude_min_km")
        if not isinstance(self.input_encoding, Mapping) or not self.input_encoding:
            errors.append("input_encoding is required")
        if not isinstance(self.scaler_contract, Mapping) or not self.scaler_contract:
            errors.append("scaler_contract is required")
        elif not {"x", "u", "a"}.issubset(set(self.scaler_contract.keys())):
            errors.append("scaler_contract must include x/u/a scaling blocks")
        if not isinstance(self.dataset_contract, Mapping) or not self.dataset_contract:
            errors.append("dataset_contract is required")
        else:
            errors.extend(
                f"dataset_contract.{key} is required"
                for key in ("target_mode", "degree_min", "degree_max")
                if self.dataset_contract.get(key) is None
            )
        if errors:
            raise ArtifactContractError("; ".join(errors))

    @property
    def is_residual(self) -> bool:
        return self.target_mode == "residual"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactContract:
        return cls(
            schema_version=_as_int(payload.get("schema_version"), ARTIFACT_CONTRACT_SCHEMA_VERSION),
            target_mode=payload.get("target_mode", ""),
            baseline_kind=payload.get("baseline_kind", "none"),
            base_degree=_as_int(payload.get("base_degree"), -1),
            target_degree=_as_int(payload.get("target_degree"), -1),
            runtime_model_kind=payload.get("runtime_model_kind", ""),
            prediction_kind=payload.get("prediction_kind", "residual_potential"),
            mu_si=_as_float(payload.get("mu_si"), MU_MOON_SI),
            r_ref_m=_as_float(payload.get("r_ref_m"), R_MOON_SI),
            a_sign=_as_float(payload.get("a_sign"), 1.0),
            altitude_min_km=payload.get("altitude_min_km"),
            altitude_max_km=payload.get("altitude_max_km"),
            input_encoding=_mapping(payload.get("input_encoding")),
            scaler_contract=_mapping(payload.get("scaler_contract")),
            dataset_contract=_mapping(payload.get("dataset_contract")),
            output_dim=_as_int(payload.get("output_dim"), 1),
            architecture_signature=payload.get("architecture_signature"),
        )

    @classmethod
    def from_resolved_config(
        cls,
        config: Mapping[str, Any],
        *,
        scaler_payload: Mapping[str, Any] | None = None,
        dataset_contract: Mapping[str, Any] | None = None,
        architecture_signature: str | None = None,
    ) -> ArtifactContract:
        if isinstance(config.get("artifact_contract"), Mapping):
            return cls.from_dict(config["artifact_contract"])
        target = TargetContract.from_resolved_config(config)
        dataset = dict(dataset_contract or _dataset_contract_from_config(config))
        if not dataset:
            raise ArtifactContractError(
                "ArtifactContract requires dataset_contract. Regenerate the ST-LRPS run "
                "with the current artifact schema."
            )
        if dataset.get("target_mode") is None:
            dataset["target_mode"] = target.target_mode
        if dataset.get("degree_min") is None:
            dataset["degree_min"] = target.base_degree
        if dataset.get("degree_max") is None:
            dataset["degree_max"] = target.target_degree
        alt_min = (
            dataset.get("altitude_min_km")
            if dataset.get("altitude_min_km") is not None
            else dataset.get("alt_min_km", config.get("altitude_min_km"))
        )
        alt_max = (
            dataset.get("altitude_max_km")
            if dataset.get("altitude_max_km") is not None
            else dataset.get("alt_max_km", config.get("altitude_max_km"))
        )
        scaler_contract = _mapping(config.get("scaler_contract"))
        if not scaler_contract and scaler_payload is not None:
            scaler_contract = _scaler_contract_from_payload(scaler_payload)
        if not scaler_contract and isinstance(config.get("scaler_summary"), Mapping):
            summary = _mapping(config.get("scaler_summary"))
            scaler_contract = {
                "schema_version": 1,
                "kind": "isometric",
                "x": {"scale": summary.get("x_scale")},
                "u": {"scale": summary.get("u_scale")},
                "a": {"scale": summary.get("a_scale")},
                "provenance": {},
            }
        runtime_kind = config.get("runtime_model_kind", "potential_autograd")
        output_dim = _as_int(config.get("output_dim"), 1)
        prediction_kind = config.get(
            "prediction_kind",
            "residual_potential" if target.is_residual else "potential",
        )
        return cls(
            schema_version=ARTIFACT_CONTRACT_SCHEMA_VERSION,
            target_mode=target.target_mode,
            baseline_kind=target.baseline_kind,
            base_degree=target.base_degree,
            target_degree=target.target_degree,
            runtime_model_kind=runtime_kind,
            prediction_kind=prediction_kind,
            mu_si=target.mu_si,
            r_ref_m=target.r_ref_m,
            a_sign=target.a_sign,
            altitude_min_km=alt_min,
            altitude_max_km=alt_max,
            input_encoding=_input_encoding_from_config(config) or {"embedding_type": "raw", "input_feature_dim": 3},
            scaler_contract=scaler_contract,
            dataset_contract=dataset,
            output_dim=output_dim,
            architecture_signature=architecture_signature or config.get("architecture_signature"),
        )

    @classmethod
    def from_benchmark_config(cls, config: Mapping[str, Any]) -> ArtifactContract:
        scenario = _mapping(config.get("scenario"))
        truth = _mapping(config.get("truth"))
        surrogate = _mapping(config.get("surrogate"))
        propagation = _mapping(config.get("propagation"))
        base_degree = _as_int(surrogate.get("baseline_degree"), -1)
        target_degree = _as_int(truth.get("degree"), -1)
        dataset = {
            "schema_version": 1,
            "dataset_kind": "benchmark_request",
            "target_mode": "residual" if bool(surrogate.get("enabled", True)) else "full",
            "degree_min": base_degree,
            "degree_max": target_degree,
            "altitude_min_km": scenario.get("altitude_min_km"),
            "altitude_max_km": scenario.get("altitude_max_km"),
            "coordinate_frame": MOON_FIXED_FRAME,
            "units": {"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"},
        }
        return cls(
            schema_version=ARTIFACT_CONTRACT_SCHEMA_VERSION,
            target_mode="residual" if bool(surrogate.get("enabled", True)) else "full",
            baseline_kind="spherical_harmonics" if base_degree >= 0 else "point_mass",
            base_degree=base_degree,
            target_degree=target_degree,
            runtime_model_kind=surrogate.get("runtime_model_kind", "potential_autograd"),
            prediction_kind="residual_potential",
            mu_si=_as_float(config.get("resolved_mu_si"), MU_MOON_SI),
            r_ref_m=_as_float(config.get("resolved_r_ref_m"), R_MOON_SI),
            a_sign=_as_float(config.get("resolved_a_sign"), 1.0),
            altitude_min_km=scenario.get("altitude_min_km"),
            altitude_max_km=scenario.get("altitude_max_km"),
            input_encoding={"benchmark_dtype": propagation.get("dtype"), "embedding_type": "artifact_runtime"},
            scaler_contract={"schema_version": 1, "kind": "benchmark_request", "x": {}, "u": {}, "a": {}},
            dataset_contract=dataset,
            output_dim=1,
            architecture_signature=None,
        )

    def compatibility_report(
        self,
        other: ArtifactContract | Mapping[str, Any],
        *,
        strict_domain: bool = False,
    ) -> dict[str, Any]:
        requested = other if isinstance(other, ArtifactContract) else ArtifactContract.from_dict(other)
        errors: list[str] = []
        warnings: list[str] = []

        def error(msg: str) -> None:
            errors.append(msg)

        def warn(msg: str) -> None:
            warnings.append(msg)

        if self.target_mode != requested.target_mode:
            error(f"target_mode mismatch: artifact={self.target_mode} requested={requested.target_mode}")
        if self.baseline_kind != requested.baseline_kind:
            error(f"baseline_kind mismatch: artifact={self.baseline_kind} requested={requested.baseline_kind}")
        if self.base_degree != requested.base_degree:
            error(
                "ST-LRPS artifact was trained as residual over "
                f"{self.baseline_kind} degree {self.base_degree}, but downstream requested "
                f"degree {requested.base_degree}."
            )
        if self.target_degree != requested.target_degree:
            error(f"target_degree mismatch: artifact={self.target_degree} requested={requested.target_degree}")
        if self.runtime_model_kind != requested.runtime_model_kind:
            error(
                f"runtime_model_kind mismatch: artifact={self.runtime_model_kind} "
                f"requested={requested.runtime_model_kind}"
            )
        if self.output_dim != requested.output_dim:
            error(f"output_dim mismatch: artifact={self.output_dim} requested={requested.output_dim}")
        if abs(self.mu_si - requested.mu_si) > 1.0:
            error(f"mu_si mismatch: artifact={self.mu_si} requested={requested.mu_si}")
        if abs(self.r_ref_m - requested.r_ref_m) > 1.0:
            error(f"r_ref_m mismatch: artifact={self.r_ref_m} requested={requested.r_ref_m}")
        if self.a_sign != requested.a_sign:
            error(f"a_sign mismatch: artifact={self.a_sign} requested={requested.a_sign}")
        if not self.scaler_contract:
            error("artifact scaler_contract is missing")
        if self.architecture_signature and requested.architecture_signature:
            if self.architecture_signature != requested.architecture_signature:
                error(
                    "architecture_signature mismatch: "
                    f"artifact={self.architecture_signature} requested={requested.architecture_signature}"
                )

        if self.altitude_min_km is None or self.altitude_max_km is None:
            warn("artifact altitude envelope is missing; domain extrapolation cannot be audited")
        elif requested.altitude_min_km is not None and requested.altitude_max_km is not None:
            outside = (
                float(requested.altitude_min_km) < float(self.altitude_min_km)
                or float(requested.altitude_max_km) > float(self.altitude_max_km)
            )
            if outside:
                msg = (
                    "benchmark altitude envelope "
                    f"[{requested.altitude_min_km}, {requested.altitude_max_km}] km exceeds "
                    f"artifact training envelope [{self.altitude_min_km}, {self.altitude_max_km}] km"
                )
                if strict_domain:
                    error(msg)
                else:
                    warn(msg)

        if not resolve_dataset_hash(self.dataset_contract):
            warn("artifact dataset_contract.content_sha256/dataset_sha256 is unavailable")
        if not self.dataset_contract.get("source_gravity_file_sha256"):
            warn("artifact dataset_contract.source_gravity_file_sha256 is unavailable")

        return {
            "compatible": not errors,
            "errors": errors,
            "warnings": warnings,
            "artifact": self.to_dict(),
            "requested": requested.to_dict(),
        }

    def require_compatible(
        self,
        other: ArtifactContract | Mapping[str, Any],
        *,
        strict: bool = True,
        strict_domain: bool = False,
    ) -> dict[str, Any]:
        report = self.compatibility_report(other, strict_domain=strict_domain)
        if strict and report["errors"]:
            raise ArtifactContractError("; ".join(report["errors"]))
        return report


# ---------------------------------------------------------------------------
# Paper-safe artifact metadata requirements (roadmap R26).
#
# ``paper_safe=True`` inference/benchmark runs refuse to start unless every
# logical field below can be resolved from the artifact (contract + resolved
# config + run_provenance + checkpoint). Research mode may load an incomplete
# legacy artifact, but only through an explicit override that is recorded in
# the runtime's metadata (``legacy_override=True`` + the missing-field list).
# ---------------------------------------------------------------------------

PAPER_SAFE_METADATA_SCHEMA_VERSION = 1

#: Logical field name -> short description of the canonical source.
PAPER_SAFE_REQUIRED_METADATA: dict[str, str] = {
    "model_kind": "run_provenance.model_kind (surrogate model identity)",
    "runtime_kind": "artifact_contract.runtime_model_kind",
    "body": "resolved config central_body (lunar-compatible)",
    "gravity_model_name": "dataset_contract.source_gravity_model",
    "gravity_model_hash": "dataset_contract.source_gravity_file_sha256",
    "baseline_degree": "artifact_contract.base_degree",
    "residual_degree_range": "artifact_contract.[base_degree, target_degree]",
    "train_altitude_min_km": "artifact_contract.altitude_min_km",
    "train_altitude_max_km": "artifact_contract.altitude_max_km",
    "scaler_source": "scaler_contract.provenance (fit provenance)",
    "x_scale": "scaler_contract.x.scale",
    "u_scale": "scaler_contract.u.scale",
    "dtype": "resolved config dtype (training/inference precision)",
    "architecture": "artifact_contract.architecture_signature",
    "parameter_count": "resolved config parameter_count",
    "training_data_hash": "dataset_contract.content_sha256 (legacy: dataset_sha256)",
    "validation_data_hash": "run_provenance.split_manifest_sha256",
    "git_commit": "run_provenance.git_commit",
    "created_at": "run_provenance.created_at_utc",
    "training_config_hash": "run_provenance.training_config_hash",
    "loss_config": "resolved config loss_config block",
    "domain_guard_policy": "resolved config domain_guard_policy",
}


def _nonempty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nonempty_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping) and value:
        return {str(k): v for k, v in value.items()}
    return None


def _positive_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0.0 else None


def collect_paper_safe_metadata(
    *,
    contract: ArtifactContract | Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    run_provenance: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve every R26 paper-safe metadata field from its canonical source.

    Returns a mapping ``logical_field -> value`` where ``None`` means the field
    could not be resolved (missing metadata). Resolution never invents values:
    each field comes only from the artifact contract, the resolved training
    config, the run provenance block, or the checkpoint payload.
    """
    art = contract if isinstance(contract, ArtifactContract) else ArtifactContract.from_dict(contract)
    cfg = dict(config or {})
    ckpt = dict(checkpoint or {})
    prov = _nonempty_mapping(run_provenance)
    if prov is None:
        prov = _nonempty_mapping(cfg.get("run_provenance")) or _nonempty_mapping(ckpt.get("run_provenance")) or {}

    dataset = _mapping(art.dataset_contract)
    scaler = _mapping(art.scaler_contract)
    scaler_x = _mapping(scaler.get("x"))
    scaler_u = _mapping(scaler.get("u"))

    degree_range: list[int] | None = None
    if art.target_degree >= 0:
        degree_range = [int(art.base_degree), int(art.target_degree)]

    parameter_count = cfg.get("parameter_count")
    if parameter_count is None:
        parameter_count = ckpt.get("parameter_count")
    try:
        parameter_count = int(parameter_count) if parameter_count is not None else None
    except (TypeError, ValueError):
        parameter_count = None
    if parameter_count is not None and parameter_count <= 0:
        parameter_count = None

    return {
        "model_kind": _nonempty_str(prov.get("model_kind")) or _nonempty_str(cfg.get("runtime_model_kind")),
        "runtime_kind": _nonempty_str(art.runtime_model_kind),
        "body": _nonempty_str(cfg.get("central_body") or dataset.get("central_body")),
        "gravity_model_name": _nonempty_str(dataset.get("source_gravity_model")),
        "gravity_model_hash": _nonempty_str(dataset.get("source_gravity_file_sha256")),
        "baseline_degree": int(art.base_degree),
        "residual_degree_range": degree_range,
        "train_altitude_min_km": art.altitude_min_km,
        "train_altitude_max_km": art.altitude_max_km,
        "scaler_source": _nonempty_mapping(scaler.get("provenance")),
        "x_scale": _positive_float(scaler_x.get("scale")),
        "u_scale": _positive_float(scaler_u.get("scale")),
        "dtype": _nonempty_str(cfg.get("dtype") or cfg.get("torch_dtype")),
        "architecture": _nonempty_str(art.architecture_signature),
        "parameter_count": parameter_count,
        "training_data_hash": _nonempty_str(resolve_dataset_hash(dataset, cfg)),
        "validation_data_hash": _nonempty_str(
            prov.get("split_manifest_sha256") or cfg.get("validation_data_hash")
        ),
        "git_commit": _nonempty_str(prov.get("git_commit")),
        "created_at": _nonempty_str(prov.get("created_at_utc") or ckpt.get("created_at_utc") or cfg.get("created_at_utc")),
        "training_config_hash": _nonempty_str(prov.get("training_config_hash") or cfg.get("training_config_hash")),
        "loss_config": _nonempty_mapping(cfg.get("loss_config")),
        "domain_guard_policy": _nonempty_str(cfg.get("domain_guard_policy")),
    }


def paper_safe_metadata_report(
    *,
    contract: ArtifactContract | Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    run_provenance: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the paper-safe metadata completeness report for an artifact.

    ``complete=True`` only when every :data:`PAPER_SAFE_REQUIRED_METADATA` field
    resolved to a value. The report is JSON-serializable and intended to be
    recorded verbatim into benchmark manifests / runtime metadata.
    """
    fields = collect_paper_safe_metadata(
        contract=contract,
        config=config,
        run_provenance=run_provenance,
        checkpoint=checkpoint,
    )
    missing = sorted(name for name in PAPER_SAFE_REQUIRED_METADATA if fields.get(name) is None)
    return {
        "schema_version": PAPER_SAFE_METADATA_SCHEMA_VERSION,
        "fields": fields,
        "missing": missing,
        "complete": not missing,
    }


def require_paper_safe_metadata(
    *,
    contract: ArtifactContract | Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    run_provenance: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    context: str = "inference",
) -> dict[str, Any]:
    """Fail closed when paper-safe required metadata is missing.

    Raises :class:`ArtifactContractError` naming every missing field. Returns
    the completeness report when the artifact is fully described, so callers
    can record it as provenance.
    """
    report = paper_safe_metadata_report(
        contract=contract,
        config=config,
        run_provenance=run_provenance,
        checkpoint=checkpoint,
    )
    if not report["complete"]:
        details = "; ".join(
            f"{name} (expected from {PAPER_SAFE_REQUIRED_METADATA[name]})" for name in report["missing"]
        )
        raise ArtifactContractError(
            f"paper_safe {context} refused: artifact metadata is incomplete. "
            f"Missing {len(report['missing'])} required field(s): {details}. "
            "Regenerate the artifact with the current training pipeline, or load it "
            "explicitly in research mode (paper_safe=False), which records a "
            "legacy metadata override instead of silently trusting the artifact."
        )
    return report


__all__ = [
    "ARTIFACT_CONTRACT_SCHEMA_VERSION",
    "ArtifactContract",
    "ArtifactContractError",
    "BASELINE_KINDS",
    "LUNAR_BODY_ALIASES",
    "PAPER_SAFE_METADATA_SCHEMA_VERSION",
    "PAPER_SAFE_REQUIRED_METADATA",
    "PREDICTION_KINDS",
    "MOON_FIXED_FRAME",
    "REQUIRED_DERIVATIVE_CONVENTION",
    "RUNTIME_MODEL_KINDS",
    "TARGET_MODES",
    "TargetContract",
    "collect_paper_safe_metadata",
    "paper_safe_metadata_report",
    "require_paper_safe_metadata",
    "resolve_dataset_hash",
]
