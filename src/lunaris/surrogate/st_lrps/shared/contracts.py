"""Explicit target contracts for ST-LRPS lunar surrogate artifacts.

The target contract separates two ideas that used to be coupled implicitly:

* the harmonic degree range of the reference/high-degree fields, and
* whether the dataset stores residual labels or full-field labels.

Old configs may omit ``target_contract``.  Use
``TargetContract.from_legacy_config`` to reconstruct the contract from the
older flat fields without silently changing the learned physics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from lunaris.surrogate.st_lrps.data.dataset_parameters import (
    MU_MOON_SI,
    R_MOON_SI,
    is_lunar_body_signature,
)


REQUIRED_DERIVATIVE_CONVENTION = "dP_dphi_corrected_v1"
LUNAR_BODY_ALIASES = frozenset({"moon", "lunar", "selene"})
TARGET_MODES = frozenset({"residual", "full"})
BASELINE_KINDS = frozenset({"none", "point_mass", "spherical_harmonics"})
ARTIFACT_CONTRACT_SCHEMA_VERSION = 1
RUNTIME_MODEL_KINDS = frozenset({"potential_autograd"})
PREDICTION_KINDS = frozenset({"potential", "residual_potential", "force", "residual_force", "total"})


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
        object.__setattr__(self, "frame", str(self.frame or "moon_fixed_cartesian").strip())
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetContract":
        return cls(
            central_body=payload.get("central_body", "moon"),
            target_mode=payload.get("target_mode", "residual"),
            base_degree=_as_int(payload.get("base_degree"), -1),
            target_degree=_as_int(payload.get("target_degree"), -1),
            baseline_kind=payload.get("baseline_kind", "none"),
            unit_system=payload.get("unit_system", "si"),
            frame=payload.get("frame", "moon_fixed_cartesian"),
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
        *,
        allow_inferred_target_mode: bool = False,
        allow_legacy_derivative_convention: bool = False,
    ) -> "TargetContract":
        target_mode = getattr(meta, "target_mode", None)
        base_degree = _as_int(getattr(meta, "degree_min", None), -1)
        if not target_mode:
            if not allow_inferred_target_mode:
                raise ValueError(
                    "Dataset metadata is missing target_mode. Regenerate the dataset "
                    "or explicitly use legacy target-mode inference."
                )
            target_mode = "residual" if base_degree >= 0 else "full"
        target_mode = _clean_str(target_mode, "residual")
        target_degree = _as_int(
            getattr(meta, "degree_max", None),
            _as_int(getattr(meta, "requested_degree", None), -1),
        )
        baseline_kind = _baseline_kind_for(target_mode, base_degree)
        deriv = getattr(meta, "derivative_convention_version", None)
        if allow_legacy_derivative_convention and deriv != REQUIRED_DERIVATIVE_CONVENTION:
            deriv = REQUIRED_DERIVATIVE_CONVENTION
        return cls(
            central_body=getattr(meta, "central_body", None) or "moon",
            target_mode=target_mode,
            base_degree=base_degree,
            target_degree=target_degree,
            baseline_kind=baseline_kind,
            unit_system=getattr(meta, "unit_system", None) or "unknown",
            frame="moon_fixed_cartesian",
            derivative_convention_version=(deriv or REQUIRED_DERIVATIVE_CONVENTION),
            a_sign=float(a_sign),
            mu_si=float(resolved_mu_si),
            r_ref_m=float(resolved_r_ref_m),
        )

    @classmethod
    def from_legacy_config(
        cls,
        config: Mapping[str, Any],
        *,
        resolved_mu_si: Optional[float] = None,
        resolved_r_ref_m: Optional[float] = None,
        a_sign: Optional[float] = None,
    ) -> "TargetContract":
        """Reconstruct a target contract from old flat config fields."""

        if isinstance(config.get("target_contract"), Mapping):
            return cls.from_dict(config["target_contract"])

        dataset_meta = config.get("dataset_meta") if isinstance(config.get("dataset_meta"), Mapping) else {}
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
            frame=config.get("frame", "moon_fixed_cartesian"),
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


def _dataset_contract_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    meta = _mapping(config.get("dataset_meta") or config.get("dataset_contract") or config.get("dataset"))
    if not meta:
        return {}
    return {
        "schema_version": int(meta.get("schema_version", 1) or 1),
        "dataset_kind": meta.get("dataset_kind", "st_lrps_spatial_cloud"),
        "dataset_name": meta.get("dataset_name") or config.get("dataset_name"),
        "dataset_sha256": meta.get("dataset_sha256") or meta.get("sha256") or config.get("dataset_hash"),
        "target_mode": meta.get("target_mode") or config.get("target_mode"),
        "baseline_kind": meta.get("baseline_kind") or config.get("baseline_kind"),
        "degree_min": meta.get("degree_min") if meta.get("degree_min") is not None else config.get("degree_min"),
        "degree_max": meta.get("degree_max") if meta.get("degree_max") is not None else config.get("degree_max"),
        "mu_si": meta.get("mu_si") or config.get("resolved_mu_si") or config.get("mu_si"),
        "r_ref_m": meta.get("r_ref_m") or config.get("resolved_r_ref_m") or config.get("r_ref_m"),
        "a_sign": meta.get("a_sign") or config.get("resolved_a_sign") or config.get("a_sign"),
        "altitude_min_km": meta.get("alt_min_km") or meta.get("altitude_min_km"),
        "altitude_max_km": meta.get("alt_max_km") or meta.get("altitude_max_km"),
        "coordinate_frame": meta.get("coordinate_frame") or meta.get("frame") or "moon_fixed_cartesian",
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
    altitude_min_km: Optional[float]
    altitude_max_km: Optional[float]
    input_encoding: dict[str, Any]
    scaler_contract: dict[str, Any]
    dataset_contract: dict[str, Any]
    architecture_signature: Optional[str] = None

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
        if self.runtime_model_kind not in RUNTIME_MODEL_KINDS:
            errors.append(
                f"unsupported runtime_model_kind={self.runtime_model_kind!r}; "
                "only 'potential_autograd' is implemented"
            )
        if self.prediction_kind not in PREDICTION_KINDS:
            errors.append(f"unsupported prediction_kind={self.prediction_kind!r}")
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
            for key in ("target_mode", "degree_min", "degree_max"):
                if self.dataset_contract.get(key) is None:
                    errors.append(f"dataset_contract.{key} is required")
        if errors:
            raise ArtifactContractError("; ".join(errors))

    @property
    def is_residual(self) -> bool:
        return self.target_mode == "residual"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactContract":
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
            architecture_signature=payload.get("architecture_signature"),
        )

    @classmethod
    def from_legacy_config(
        cls,
        config: Mapping[str, Any],
        *,
        scaler_payload: Optional[Mapping[str, Any]] = None,
        dataset_contract: Optional[Mapping[str, Any]] = None,
        architecture_signature: Optional[str] = None,
    ) -> "ArtifactContract":
        if isinstance(config.get("artifact_contract"), Mapping):
            return cls.from_dict(config["artifact_contract"])
        target = TargetContract.from_legacy_config(config)
        dataset = dict(dataset_contract or _dataset_contract_from_config(config))
        if not dataset:
            dataset = {
                "schema_version": 1,
                "dataset_kind": "legacy_unknown",
                "target_mode": target.target_mode,
                "degree_min": target.base_degree,
                "degree_max": target.target_degree,
                "mu_si": target.mu_si,
                "r_ref_m": target.r_ref_m,
                "a_sign": target.a_sign,
                "altitude_min_km": config.get("altitude_min_km"),
                "altitude_max_km": config.get("altitude_max_km"),
                "coordinate_frame": target.frame,
                "units": {"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"},
            }
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
        return cls(
            schema_version=ARTIFACT_CONTRACT_SCHEMA_VERSION,
            target_mode=target.target_mode,
            baseline_kind=target.baseline_kind,
            base_degree=target.base_degree,
            target_degree=target.target_degree,
            runtime_model_kind=config.get("runtime_model_kind", "potential_autograd"),
            prediction_kind=config.get("prediction_kind", "residual_potential" if target.is_residual else "potential"),
            mu_si=target.mu_si,
            r_ref_m=target.r_ref_m,
            a_sign=target.a_sign,
            altitude_min_km=alt_min,
            altitude_max_km=alt_max,
            input_encoding=_input_encoding_from_config(config) or {"embedding_type": "raw", "input_feature_dim": 3},
            scaler_contract=scaler_contract,
            dataset_contract=dataset,
            architecture_signature=architecture_signature or config.get("architecture_signature"),
        )

    @classmethod
    def from_benchmark_config(cls, config: Mapping[str, Any]) -> "ArtifactContract":
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
            "coordinate_frame": "moon_fixed_cartesian",
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
            architecture_signature=None,
        )

    def compatibility_report(
        self,
        other: "ArtifactContract | Mapping[str, Any]",
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

        for key in ("dataset_sha256", "source_gravity_file_sha256"):
            if not self.dataset_contract.get(key):
                warn(f"artifact dataset_contract.{key} is unavailable")

        return {
            "compatible": not errors,
            "errors": errors,
            "warnings": warnings,
            "artifact": self.to_dict(),
            "requested": requested.to_dict(),
        }

    def require_compatible(
        self,
        other: "ArtifactContract | Mapping[str, Any]",
        *,
        strict: bool = True,
        strict_domain: bool = False,
    ) -> dict[str, Any]:
        report = self.compatibility_report(other, strict_domain=strict_domain)
        if strict and report["errors"]:
            raise ArtifactContractError("; ".join(report["errors"]))
        return report


__all__ = [
    "ARTIFACT_CONTRACT_SCHEMA_VERSION",
    "ArtifactContract",
    "ArtifactContractError",
    "BASELINE_KINDS",
    "LUNAR_BODY_ALIASES",
    "PREDICTION_KINDS",
    "REQUIRED_DERIVATIVE_CONVENTION",
    "RUNTIME_MODEL_KINDS",
    "TARGET_MODES",
    "TargetContract",
]
