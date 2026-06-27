"""Validated gravity data pack for dynamics RHS construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunaris.common.type_defs import F64Array
from lunaris.core.dynamics.requirements import _as_f64_c


@dataclass(frozen=True, slots=True, kw_only=True)
class _GravPack:
    """Engine-internal gravity pack (validated, float64, C-contiguous; GravityModel)."""
    nmax: int
    r_ref_m: float
    gm_m3s2: float

    Cnm: F64Array
    Snm: F64Array
    diag: F64Array
    subdiag: F64Array
    A: F64Array
    B: F64Array
    scale_m: F64Array

    ws_P: F64Array
    ws_dP: F64Array
    ws_cos_m: F64Array
    ws_sin_m: F64Array

    # Optional runtime degree policy for adaptive SH evaluation.
    adaptive_enabled: bool = False
    adaptive_mode: int = 0  # 0=off, 1=table, 2=power-law
    adaptive_power: float = 2.5
    adaptive_min_degree: int = 4
    adaptive_quantization_step: int = 10
    adaptive_table_alt_km: F64Array | None = None
    adaptive_table_degree: np.ndarray | None = None
    adaptive_table_len: int = 0

    def __post_init__(self) -> None:
        if self.nmax < 0:
            raise ValueError(f"nmax must be >= 0, got {self.nmax}")
        if self.r_ref_m <= 0.0 or self.gm_m3s2 <= 0.0:
            raise ValueError(
                f"r_ref_m and gm_m3s2 must be positive (r={self.r_ref_m}, gm={self.gm_m3s2})"
            )

        Cnm = _as_f64_c(self.Cnm, "Cnm")
        Snm = _as_f64_c(self.Snm, "Snm")
        scale_m = _as_f64_c(self.scale_m, "scale_m")

        if Cnm.ndim < 2 or Snm.ndim < 2:
            raise ValueError("Cnm/Snm must be at least 2D arrays.")
        if Cnm.shape[0] < self.nmax + 1 or Cnm.shape[1] < self.nmax + 1:
            raise ValueError(f"Cnm too small for nmax={self.nmax}: {Cnm.shape}")
        if Snm.shape[0] < self.nmax + 1 or Snm.shape[1] < self.nmax + 1:
            raise ValueError(f"Snm too small for nmax={self.nmax}: {Snm.shape}")
        if scale_m.ndim != 1 or scale_m.shape[0] < self.nmax + 1:
            raise ValueError(f"scale_m must be 1D len>=nmax+1, got {scale_m.shape}")

        object.__setattr__(self, "Cnm", Cnm)
        object.__setattr__(self, "Snm", Snm)
        object.__setattr__(self, "diag", _as_f64_c(self.diag, "diag"))
        object.__setattr__(self, "subdiag", _as_f64_c(self.subdiag, "subdiag"))
        object.__setattr__(self, "A", _as_f64_c(self.A, "A"))
        object.__setattr__(self, "B", _as_f64_c(self.B, "B"))
        object.__setattr__(self, "scale_m", scale_m)

        object.__setattr__(self, "ws_P", _as_f64_c(self.ws_P, "ws_P"))
        object.__setattr__(self, "ws_dP", _as_f64_c(self.ws_dP, "ws_dP"))
        object.__setattr__(self, "ws_cos_m", _as_f64_c(self.ws_cos_m, "ws_cos_m"))
        object.__setattr__(self, "ws_sin_m", _as_f64_c(self.ws_sin_m, "ws_sin_m"))

        if self.ws_cos_m.ndim != 1 or self.ws_sin_m.ndim != 1:
            raise ValueError("ws_cos_m/ws_sin_m must be 1D.")
        if self.ws_cos_m.shape[0] < self.nmax + 1 or self.ws_sin_m.shape[0] < self.nmax + 1:
            raise ValueError(f"workspace sin/cos too small for nmax={self.nmax}")

        alt_km = (
            np.zeros(1, dtype=np.float64)
            if self.adaptive_table_alt_km is None
            else _as_f64_c(self.adaptive_table_alt_km, "adaptive_table_alt_km")
        )
        deg = (
            np.zeros(1, dtype=np.int64)
            if self.adaptive_table_degree is None
            else np.ascontiguousarray(np.asarray(self.adaptive_table_degree, dtype=np.int64))
        )

        if alt_km.ndim != 1:
            raise ValueError("adaptive_table_alt_km must be 1D.")
        if deg.ndim != 1:
            raise ValueError("adaptive_table_degree must be 1D.")
        if alt_km.shape[0] != deg.shape[0]:
            raise ValueError("adaptive_table_alt_km and adaptive_table_degree must have the same length.")

        table_len = int(self.adaptive_table_len)
        if table_len < 0:
            raise ValueError(f"adaptive_table_len must be >= 0, got {table_len}")
        if table_len > int(alt_km.shape[0]):
            raise ValueError("adaptive_table_len exceeds the provided adaptive table storage.")

        object.__setattr__(self, "adaptive_power", float(self.adaptive_power))
        object.__setattr__(self, "adaptive_min_degree", int(self.adaptive_min_degree))
        object.__setattr__(self, "adaptive_quantization_step", max(1, int(self.adaptive_quantization_step)))
        object.__setattr__(self, "adaptive_table_alt_km", alt_km)
        object.__setattr__(self, "adaptive_table_degree", deg)
        object.__setattr__(self, "adaptive_table_len", table_len)

__all__ = ["_GravPack"]
