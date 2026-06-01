"""Configuration for Perturbation Budget Analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _float_list(values: Iterable[float]) -> List[float]:
    out = [float(v) for v in values]
    if not out:
        raise ValueError("list cannot be empty")
    return out


def _int_list(values: Iterable[int]) -> List[int]:
    out = sorted({int(v) for v in values})
    if not out:
        raise ValueError("list cannot be empty")
    if any(v < 0 for v in out):
        raise ValueError("degrees must be non-negative")
    return out


@dataclass(frozen=True, slots=True)
class PerturbationBudgetConfig:
    """User-facing configuration for the analysis.

    The default run is intentionally deterministic and self-contained. If
    ``gravity_model_path`` is omitted, a synthetic gravity coefficient set is
    used only so the tool and tests can run without external GRAIL data. Reports
    label this clearly; production studies should pass a real gravity model.
    """

    altitudes_km: List[float] = field(default_factory=lambda: [50.0, 100.0, 300.0, 1000.0, 3000.0])
    inclinations_deg: List[float] = field(default_factory=lambda: [0.0, 30.0, 60.0, 90.0])
    true_anomalies_deg: List[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    epochs_utc: List[str] = field(
        default_factory=lambda: [
            "2026-01-01T00:00:00Z",
            "2026-04-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        ]
    )
    sh_degrees: List[int] = field(default_factory=lambda: [20, 30, 60, 100, 200])
    reference_sh_degree: int = 200

    include_srp: bool = True
    include_albedo: bool = True
    include_thermal_ir: bool = True
    include_tides: bool = True
    include_third_body: bool = True
    include_relativity: bool = True
    include_earth_j2: bool = True

    use_ephemeris: bool = False
    use_synthetic_geometry_fallback: bool = True
    gravity_model_path: Optional[str] = None

    spacecraft_area_m2: float = 2.0
    spacecraft_mass_kg: float = 1000.0
    srp_coefficient: float = 1.8

    albedo_const: float = 0.12
    albedo_facet_lat_count: int = 6
    albedo_facet_lon_count: int = 12
    thermal_temperature_K: float = 250.0
    thermal_facet_lat_count: int = 6
    thermal_facet_lon_count: int = 12

    srp_uncertainty: float = 0.20
    area_uncertainty: float = 0.10
    mass_uncertainty: float = 0.01
    albedo_uncertainty: float = 0.30
    thermal_uncertainty: float = 0.30
    thermal_temperature_uncertainty_K: float = 10.0
    tide_uncertainty: float = 0.20
    third_body_uncertainty: float = 0.0
    relativity_uncertainty: float = 0.0

    recommendation_absolute_threshold_m_s2: float = 1.0e-12
    recommendation_uncertainty_fraction: float = 0.5
    rss_uncertainty_combination: bool = True

    output_dir: str = "outputs/perturbation_budget/default"
    random_seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "altitudes_km", _float_list(self.altitudes_km))
        object.__setattr__(self, "inclinations_deg", _float_list(self.inclinations_deg))
        object.__setattr__(self, "true_anomalies_deg", _float_list(self.true_anomalies_deg))
        object.__setattr__(self, "epochs_utc", [str(e) for e in self.epochs_utc] or ["synthetic-epoch-0"])
        object.__setattr__(self, "sh_degrees", _int_list(self.sh_degrees))
        if int(self.reference_sh_degree) not in self.sh_degrees:
            object.__setattr__(self, "reference_sh_degree", max(self.sh_degrees))
        if any(a < 0.0 for a in self.altitudes_km):
            raise ValueError("altitudes_km must be non-negative")
        if self.spacecraft_mass_kg <= 0.0:
            raise ValueError("spacecraft_mass_kg must be > 0")
        if self.spacecraft_area_m2 < 0.0:
            raise ValueError("spacecraft_area_m2 must be >= 0")
        if self.recommendation_absolute_threshold_m_s2 < 0.0:
            raise ValueError("recommendation_absolute_threshold_m_s2 must be >= 0")
        if self.recommendation_uncertainty_fraction < 0.0:
            raise ValueError("recommendation_uncertainty_fraction must be >= 0")

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_csv_floats(value: str) -> List[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_on_off(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected on/off boolean, got {value!r}")
