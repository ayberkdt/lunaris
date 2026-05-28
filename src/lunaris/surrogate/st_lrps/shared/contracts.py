"""Explicit target contracts for ST-LRPS lunar surrogate artifacts.

The target contract separates two ideas that used to be coupled implicitly:

* the harmonic degree range of the reference/high-degree fields, and
* whether the dataset stores residual labels or full-field labels.

Old configs may omit ``target_contract``.  Use
``TargetContract.from_legacy_config`` to reconstruct the contract from the
older flat fields without silently changing the learned physics.
"""

from __future__ import annotations

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


__all__ = [
    "BASELINE_KINDS",
    "LUNAR_BODY_ALIASES",
    "REQUIRED_DERIVATIVE_CONVENTION",
    "TARGET_MODES",
    "TargetContract",
]
