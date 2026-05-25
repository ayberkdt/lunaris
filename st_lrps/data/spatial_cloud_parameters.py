# -*- coding: utf-8 -*-
"""
Spatial Cloud Parameters
========================

Configuration SSOT for the surrogate gravity point-cloud generator.

This module stores *sampling* and *output* choices only. Physical constants
(lunar GM, reference radius, gravity model path) live in
:mod:`st_lrps.dataset_parameters`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from st_lrps.data.dataset_parameters import DEFAULT_DATASET_CONFIG


# =============================================================================
# 1.                      SAMPLING ENUMS / HELPERS
# =============================================================================


class SamplingStrategy(str, Enum):
    """
    Radial sampling law for the lunar spherical shell.

    Why this matters
    ----------------
    PINN/ST-LRPS gravity surrogates are sensitive to where the training points land.
    A volumetrically uniform shell over-emphasizes the high-altitude region,
    where the gravity field is smoother and easier. The near-surface band holds
    the sharper harmonics and is therefore the most valuable region for
    learning ``dU`` and ``∇dU``.

    Strategy summary
    ----------------
    ``uniform``
        Homogeneous in shell volume. Good for generic diagnostics, but it can
        starve the lower-altitude band when the shell is thick.

    ``inverse_r2``
        Surface-biased. Draws more points near the lunar reference radius, which
        improves residual-field learning for higher SH degrees.

    ``mixed``
        Blend of ``uniform`` and ``inverse_r2``. This is the recommended
        default because it preserves broad-shell coverage while still feeding
        the network enough difficult near-surface samples.
    """

    UNIFORM = "uniform"
    INVERSE_R2 = "inverse_r2"
    MIXED = "mixed"


# =============================================================================
# 2.                         CONFIGURATION DATACLASS
# =============================================================================


@dataclass(frozen=True)
class SpatialCloudConfig:
    """
    Immutable sampling plan for generating ``[x, y, z, U, ax, ay, az]`` clouds.

    Generates ``[x, y, z, ΔU, Δax, Δay, Δaz]`` residual clouds for SH degrees
    in the range ``(degree_min, degree_max]``.  Physical constants and the GFC
    model path are inherited from :mod:`st_lrps.dataset_parameters`.
    """

    # ----------------------------
    # Gravity field source
    # ----------------------------
    degree_max: int = 100
    degree_min: int = 20  # Network learns residual for degrees (degree_min, degree_max]. -1 = full field.
    gfc_path: Optional[str] = None  # None => DEFAULT_DATASET_CONFIG.gravity_gfc_path

    # ----------------------------
    # Spatial sampling envelope
    # ----------------------------
    n_samples: int = 2_000_000
    alt_min_km: float = 200.0
    alt_max_km: float = 600.0
    sampling_strategy: str = SamplingStrategy.MIXED.value
    surface_bias_ratio: float = 0.70

    # ----------------------------
    # Throughput / scaling
    # ----------------------------
    chunk_size: int = 50_000
    workers: int = max(1, os.cpu_count() or 1)
    no_multiprocessing: bool = False

    # ----------------------------
    # Output contract
    # ----------------------------
    out_format: str = "h5"  # "h5" | "pt"
    out_path: str = ""      # empty => generated automatically
    dtype: str = "float32"  # "float32" | "float64"
    canonical: bool = False

    # ----------------------------
    # Reproducibility
    # ----------------------------
    seed: int = 12345

    def __post_init__(self) -> None:
        if int(self.degree_max) < 0:
            raise ValueError("degree_max must be >= 0")
        if int(self.degree_min) < -1:
            raise ValueError("degree_min must be >= -1")
        if int(self.degree_max) <= int(self.degree_min):
            raise ValueError("degree_max must be > degree_min")
        if int(self.n_samples) <= 0:
            raise ValueError("n_samples must be > 0")
        if float(self.alt_min_km) < 0.0:
            raise ValueError("alt_min_km must be >= 0")
        if float(self.alt_max_km) <= float(self.alt_min_km):
            raise ValueError("alt_max_km must be > alt_min_km")
        valid_strategies = {item.value for item in SamplingStrategy}
        if str(self.sampling_strategy) not in valid_strategies:
            raise ValueError(f"sampling_strategy must be one of {sorted(valid_strategies)}")
        if not (0.0 <= float(self.surface_bias_ratio) <= 1.0):
            raise ValueError("surface_bias_ratio must be in [0, 1]")
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be > 0")
        if int(self.workers) <= 0:
            raise ValueError("workers must be > 0")
        if self.out_format not in ("h5", "pt"):
            raise ValueError("out_format must be 'h5' or 'pt'")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")

    def resolved_gfc_path(self) -> str:
        """
        Return the gravity model path that should actually be loaded.

        This keeps CLI overrides and JSON round-trips deterministic.
        """

        if self.gfc_path and str(self.gfc_path).strip():
            return str(self.gfc_path)
        return str(getattr(DEFAULT_DATASET_CONFIG, "gravity_gfc_path"))

    def resolved_out_path(self) -> str:
        """
        Return the effective output filename.

        The name includes ``moon`` so legacy Earth clouds are visually distinct
        on disk and harder to confuse during manual inspection.
        """

        if self.out_path and str(self.out_path).strip():
            return str(self.out_path)

        suffix = ".h5" if self.out_format == "h5" else ".pt"
        return (
            "potential_cloud_moon_"
            f"deg{int(self.degree_min)}to{int(self.degree_max)}_"
            f"alt{int(round(float(self.alt_min_km)))}to{int(round(float(self.alt_max_km)))}"
            f"{suffix}"
        )

    @property
    def coeff_source(self) -> str:
        """
        Backward-compatible label for the gravity coefficient source.

        Older regression tests and notes used ``coeff_source`` as a plain string
        even though the current SSOT resolves the actual file path elsewhere.
        """

        return "gfc"

    def to_dict(self) -> dict:
        """Return a stable, JSON-serializable provenance mapping."""

        return {
            "degree_max": int(self.degree_max),
            "degree_min": int(self.degree_min),
            "gfc_path": self.resolved_gfc_path(),
            "n_samples": int(self.n_samples),
            "alt_min_km": float(self.alt_min_km),
            "alt_max_km": float(self.alt_max_km),
            "sampling_strategy": str(self.sampling_strategy),
            "surface_bias_ratio": float(self.surface_bias_ratio),
            "chunk_size": int(self.chunk_size),
            "workers": int(self.workers),
            "no_multiprocessing": bool(self.no_multiprocessing),
            "out_format": str(self.out_format),
            "out_path": self.resolved_out_path(),
            "dtype": str(self.dtype),
            "canonical": bool(self.canonical),
            "seed": int(self.seed),
            "central_body": "moon",
            "dataset_gravity_expected_norm": str(getattr(DEFAULT_DATASET_CONFIG, "gravity_expected_norm", "")),
            "dataset_gravity_strict_norm": bool(getattr(DEFAULT_DATASET_CONFIG, "gravity_strict_norm", True)),
        }

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Write the configuration to a JSON file for experiment tracking."""

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=indent, sort_keys=True)

    @staticmethod
    def from_json(path: str | Path) -> "SpatialCloudConfig":
        """Load a config from JSON while ignoring extra provenance fields."""

        in_path = Path(path)
        with in_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        return SpatialCloudConfig(
            degree_max=int(payload.get("degree_max", 100)),
            degree_min=int(payload.get("degree_min", 20)),
            gfc_path=payload.get("gfc_path", None),
            n_samples=int(payload.get("n_samples", 500_000)),
            alt_min_km=float(payload.get("alt_min_km", 200.0)),
            alt_max_km=float(payload.get("alt_max_km", 600.0)),
            sampling_strategy=str(payload.get("sampling_strategy", SamplingStrategy.MIXED.value)),
            surface_bias_ratio=float(payload.get("surface_bias_ratio", 0.70)),
            chunk_size=int(payload.get("chunk_size", 50_000)),
            workers=int(payload.get("workers", max(1, os.cpu_count() or 1))),
            no_multiprocessing=bool(payload.get("no_multiprocessing", False)),
            out_format=str(payload.get("out_format", "h5")),
            out_path=str(payload.get("out_path", "")),
            dtype=str(payload.get("dtype", "float32")),
            canonical=bool(payload.get("canonical", False)),
            seed=int(payload.get("seed", 12345)),
        )

    def to_cli_args(self) -> List[str]:
        """Convert the config into CLI arguments for the generator script."""

        args: List[str] = [
            "--degree-max", str(int(self.degree_max)),
            "--degree-min", str(int(self.degree_min)),
            "--n-samples", str(int(self.n_samples)),
            "--alt-range", str(float(self.alt_min_km)), str(float(self.alt_max_km)),
            "--sampling-strategy", str(self.sampling_strategy),
            "--surface-bias-ratio", str(float(self.surface_bias_ratio)),
            "--chunk-size", str(int(self.chunk_size)),
            "--workers", str(int(self.workers)),
            "--format", str(self.out_format),
            "--out", self.resolved_out_path(),
            "--dtype", str(self.dtype),
            "--seed", str(int(self.seed)),
        ]

        args += ["--gfc-path", self.resolved_gfc_path()]
        if self.canonical:
            args.append("--canonical")
        if self.no_multiprocessing:
            args.append("--no-multiprocessing")
        return args


DEFAULT_SPATIAL_CLOUD_CONFIG: SpatialCloudConfig = SpatialCloudConfig()


# -----------------------------------------------------------------------------
# Backward-compatible preset surface
# -----------------------------------------------------------------------------
# Some older utilities and tests still expect a small preset registry.  We keep
# it intentionally compact and Moon-only so Earth-era names cannot sneak back.
PRESETS = {
    "moon_residual_balanced": DEFAULT_SPATIAL_CLOUD_CONFIG,
    "moon_residual_surface_focus": SpatialCloudConfig(
        sampling_strategy=SamplingStrategy.INVERSE_R2.value,
        surface_bias_ratio=1.0,
    ),
    "debug_moon_quick": SpatialCloudConfig(
        n_samples=25_000,
        chunk_size=10_000,
        workers=1,
    ),
}


# =============================================================================
# 3.                   DATASET SUITE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CloudSuiteConfig:
    """
    Immutable configuration for a multi-file dataset suite.

    A suite produces a folder containing:
    - train_hybrid.h5        (stratified + inverse_r2 + residual_mag + boundary)
    - val_uniform.h5
    - test_uniform.h5
    - ood_low.h5
    - ood_high.h5
    - ood_combined.h5
    - manifest.json

    Physical constants and the GFC model path are inherited from
    :mod:`st_lrps.dataset_parameters`.
    """

    # ----------------------------
    # Gravity field source
    # ----------------------------
    degree_min: int = 20
    degree_max: int = 100
    gfc_path: Optional[str] = None  # None => DEFAULT_DATASET_CONFIG.gravity_gfc_path

    # ----------------------------
    # Spatial sampling envelope
    # ----------------------------
    train_alt_min_km: float = 200.0
    train_alt_max_km: float = 600.0
    ood_margin_km: float = 40.0

    # ----------------------------
    # Train hybrid component allocation
    # ----------------------------
    train_stratified_uniform_n: int = 2_000_000
    train_inverse_r2_n: int = 1_000_000
    train_residual_mag_n: int = 1_000_000
    train_boundary_n: int = 1_000_000

    # ----------------------------
    # Independent validation/test/OOD
    # ----------------------------
    val_n: int = 1_000_000
    test_n: int = 1_000_000
    ood_low_n: int = 250_000
    ood_high_n: int = 250_000

    # ----------------------------
    # Reproducibility seeds
    # ----------------------------
    base_seed: int = 42
    train_uniform_seed: int = 42
    train_inverse_r2_seed: int = 142
    train_residual_mag_seed: int = 242
    train_boundary_seed: int = 342
    val_seed: int = 1042
    test_seed: int = 2042
    ood_low_seed: int = 3042
    ood_high_seed: int = 4042

    # ----------------------------
    # Residual-magnitude weighted sampling
    # ----------------------------
    residual_mag_candidate_multiplier: int = 5
    residual_mag_weight_power: float = 0.5
    residual_mag_probability_floor: float = 1e-3
    # Memory-bounded streaming (weighted reservoir) residual-mag sampling.
    # True keeps peak memory at O(n_bin) instead of O(n_bin * multiplier);
    # False reproduces the exact legacy in-memory method bit-for-bit.
    residual_mag_streaming: bool = True

    # ----------------------------
    # Boundary buffer
    # ----------------------------
    boundary_mode: str = "strict"   # "strict" or "soft"
    boundary_width_km: float = 20.0

    # ----------------------------
    # Output
    # ----------------------------
    out_format: str = "h5"
    chunk_size: int = 50_000
    dtype: str = "float32"
    workers: int = 1
    suite_name: str = ""

    def __post_init__(self) -> None:
        if int(self.degree_max) <= int(self.degree_min):
            raise ValueError("degree_max must be > degree_min")
        if float(self.train_alt_max_km) <= float(self.train_alt_min_km):
            raise ValueError("train_alt_max_km must be > train_alt_min_km")
        if float(self.ood_margin_km) <= 0.0:
            raise ValueError("ood_margin_km must be > 0")
        if self.boundary_mode not in ("strict", "soft"):
            raise ValueError("boundary_mode must be 'strict' or 'soft'")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")

    @property
    def ood_low_alt_min_km(self) -> float:
        return float(self.train_alt_min_km) - float(self.ood_margin_km)

    @property
    def ood_low_alt_max_km(self) -> float:
        return float(self.train_alt_min_km)

    @property
    def ood_high_alt_min_km(self) -> float:
        return float(self.train_alt_max_km)

    @property
    def ood_high_alt_max_km(self) -> float:
        return float(self.train_alt_max_km) + float(self.ood_margin_km)

    @property
    def train_total_n(self) -> int:
        return (
            int(self.train_stratified_uniform_n)
            + int(self.train_inverse_r2_n)
            + int(self.train_residual_mag_n)
            + int(self.train_boundary_n)
        )

    def resolved_gfc_path(self) -> str:
        if self.gfc_path and str(self.gfc_path).strip():
            return str(self.gfc_path)
        return str(getattr(DEFAULT_DATASET_CONFIG, "gravity_gfc_path"))

    def to_dict(self) -> dict:
        import os as _os
        return {
            "degree_min": int(self.degree_min),
            "degree_max": int(self.degree_max),
            "gfc_path": self.resolved_gfc_path(),
            "train_alt_min_km": float(self.train_alt_min_km),
            "train_alt_max_km": float(self.train_alt_max_km),
            "ood_margin_km": float(self.ood_margin_km),
            "ood_low_alt_min_km": float(self.ood_low_alt_min_km),
            "ood_low_alt_max_km": float(self.ood_low_alt_max_km),
            "ood_high_alt_min_km": float(self.ood_high_alt_min_km),
            "ood_high_alt_max_km": float(self.ood_high_alt_max_km),
            "train_stratified_uniform_n": int(self.train_stratified_uniform_n),
            "train_inverse_r2_n": int(self.train_inverse_r2_n),
            "train_residual_mag_n": int(self.train_residual_mag_n),
            "train_boundary_n": int(self.train_boundary_n),
            "train_total_n": int(self.train_total_n),
            "val_n": int(self.val_n),
            "test_n": int(self.test_n),
            "ood_low_n": int(self.ood_low_n),
            "ood_high_n": int(self.ood_high_n),
            "base_seed": int(self.base_seed),
            "train_uniform_seed": int(self.train_uniform_seed),
            "train_inverse_r2_seed": int(self.train_inverse_r2_seed),
            "train_residual_mag_seed": int(self.train_residual_mag_seed),
            "train_boundary_seed": int(self.train_boundary_seed),
            "val_seed": int(self.val_seed),
            "test_seed": int(self.test_seed),
            "ood_low_seed": int(self.ood_low_seed),
            "ood_high_seed": int(self.ood_high_seed),
            "residual_mag_candidate_multiplier": int(self.residual_mag_candidate_multiplier),
            "residual_mag_weight_power": float(self.residual_mag_weight_power),
            "residual_mag_probability_floor": float(self.residual_mag_probability_floor),
            "residual_mag_streaming": bool(self.residual_mag_streaming),
            "boundary_mode": str(self.boundary_mode),
            "boundary_width_km": float(self.boundary_width_km),
            "out_format": str(self.out_format),
            "chunk_size": int(self.chunk_size),
            "dtype": str(self.dtype),
            "workers": int(self.workers),
            "suite_name": str(self.suite_name),
            "central_body": "moon",
        }


DEFAULT_CLOUD_SUITE_CONFIG: CloudSuiteConfig = CloudSuiteConfig()


# ---------------------------------------------------------------------------
# Suite presets
# ---------------------------------------------------------------------------
SUITE_PRESETS: "Dict[str, CloudSuiteConfig]" = {
    "debug_suite": CloudSuiteConfig(
        train_stratified_uniform_n=50_000,
        train_inverse_r2_n=20_000,
        train_residual_mag_n=20_000,
        train_boundary_n=10_000,
        val_n=20_000,
        test_n=20_000,
        ood_low_n=10_000,
        ood_high_n=10_000,
        suite_name="debug_suite",
    ),
    "baseline_uniform_suite": CloudSuiteConfig(
        train_stratified_uniform_n=2_000_000,
        train_inverse_r2_n=0,
        train_residual_mag_n=0,
        train_boundary_n=0,
        val_n=500_000,
        test_n=1_000_000,
        ood_low_n=250_000,
        ood_high_n=250_000,
        suite_name="baseline_uniform_suite",
    ),
    "recommended_hybrid_5M": CloudSuiteConfig(
        train_stratified_uniform_n=2_000_000,
        train_inverse_r2_n=1_000_000,
        train_residual_mag_n=1_000_000,
        train_boundary_n=1_000_000,
        val_n=1_000_000,
        test_n=1_000_000,
        ood_low_n=250_000,
        ood_high_n=250_000,
        suite_name="recommended_hybrid_5M",
    ),
    "high_accuracy_10M": CloudSuiteConfig(
        train_stratified_uniform_n=4_000_000,
        train_inverse_r2_n=2_000_000,
        train_residual_mag_n=2_000_000,
        train_boundary_n=2_000_000,
        val_n=2_000_000,
        test_n=2_000_000,
        ood_low_n=500_000,
        ood_high_n=500_000,
        suite_name="high_accuracy_10M",
    ),
}


__all__ = [
    "SamplingStrategy",
    "SpatialCloudConfig",
    "DEFAULT_SPATIAL_CLOUD_CONFIG",
    "PRESETS",
    "CloudSuiteConfig",
    "DEFAULT_CLOUD_SUITE_CONFIG",
    "SUITE_PRESETS",
]
