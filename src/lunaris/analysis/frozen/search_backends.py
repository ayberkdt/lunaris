"""Concrete propagation backends for the frozen-orbit search pipeline.

Two adapters wire the injected pipeline protocols to real Lunaris paths:

- :class:`TorchSHScreeningPropagator` — batch screening on the shared R07
  fixed-step loop via :class:`~lunaris.core.torch_sh_propagator.TorchSHBatchPropagator`
  (classical SH physics, CPU or CUDA; honest backend naming, VRAM-aware
  chunking, recorded provenance).
- :class:`ClassicalSHValidationPropagator` — single-orbit CPU reference
  propagation (adaptive DOP853 through :func:`lunaris.core.propagation.propagator.propagate`),
  the only backend whose label satisfies the R21 classical-SH rule.

Both keep torch/Numba imports lazy so ``lunaris.analysis.frozen`` stays
importable without the optional heavy dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEG2RAD = np.pi / 180.0


def _load_gravity_model(gravity_file: str | None, degree: int) -> Any:
    """Load the classical SH gravity model used by both adapters."""
    from lunaris.physics.spherical_harmonics import GravityModel

    if gravity_file is None:
        from lunaris.core.config import load_default_config

        gravity_file = str(load_default_config().gravity.file_path)
    return GravityModel.from_file(path=str(gravity_file), requested_degree=int(degree))


def _gravity_provenance(model: Any, degree: int, gravity_file: str | None) -> dict[str, Any]:
    return {
        "gravity_model": {
            "name": str(getattr(model, "name", "unknown")),
            "file": str(gravity_file) if gravity_file else "default_config",
            "degree": int(degree),
            "loaded_degree_max": int(getattr(model, "degree_max", 0)),
        },
        "third_body": {"earth": False, "sun": False},
        "force_model_scope": "gravity_only",
        "frame": "identity (gravity field fixed in the integration frame)",
    }


class TorchSHScreeningPropagator:
    """Stage-1 batch screening on the torch classical-SH backend (R07 loop)."""

    def __init__(
        self,
        *,
        degree: int = 8,
        dt_s: float = 60.0,
        device: str = "auto",
        gravity_file: str | None = None,
        chunk_size: int | None = None,
    ) -> None:
        import torch

        from lunaris.common.batch_defs import BatchPropagationConfig
        from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
        from lunaris.core.dynamics import DynamicsEngine
        from lunaris.core.torch_sh_propagator import TorchSHBatchPropagator

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        model = _load_gravity_model(gravity_file, degree)
        engine = DynamicsEngine(
            sc_props=SpacecraftProps(mass_kg=100.0, area_m2=1.0, cr=1.3),
            flags=PerturbationFlags(enable_sh=True),
            gravity_model=model,
            ephem_manager=None,
            surface_provider=None,
            earth_j2=None,
            allow_identity_rotation=True,
        )
        batch_cfg = BatchPropagationConfig(
            n_samples=2,  # not used by direct propagate(Y0, ...) calls
            seed=0,
            sh_degree=int(degree),
            dt_s=float(dt_s),
            torch_dtype="float64",
            impact_alt_km=0.0,
        )
        self._propagator = TorchSHBatchPropagator(
            engine,
            batch_cfg,
            PerturbationFlags(enable_sh=True),
            device=self._device,
            chunk_size=chunk_size,
        )
        self.backend_name = (
            "torch_cuda_sh" if self._device.type == "cuda" else "torch_cpu_sh"
        )
        self.provenance = {
            **_gravity_provenance(model, degree, gravity_file),
            "backend": self.backend_name,
            "device": str(self._device),
            "dtype": "float64",
            "dt_s": float(dt_s),
            "loop": "run_batched_fixed_step (R07)",
        }

    def propagate(
        self, Y0: np.ndarray, duration_s: float, output_dt_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = int(np.asarray(Y0).shape[0])
        ones = np.ones(n, dtype=np.float64)
        t_out, Y_out, impact_flags, t_impact = self._propagator.propagate(
            np.asarray(Y0, dtype=np.float64),
            ones * 100.0,
            ones,
            ones * 2.2,
            ones * 1.3,
            float(duration_s),
            float(output_dt_s),
        )
        return t_out, Y_out, impact_flags, t_impact


class ClassicalSHValidationPropagator:
    """Stage-3 single-orbit classical SH validation on the CPU reference path."""

    def __init__(
        self,
        *,
        degree: int = 50,
        gravity_file: str | None = None,
        rtol: float = 1e-10,
        atol: float = 1e-9,
        method: str = "DOP853",
        third_body: bool = False,
    ) -> None:
        from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
        from lunaris.core.dynamics import DynamicsEngine

        model = _load_gravity_model(gravity_file, degree)
        # Gravity-only validation config: third-body needs an ephemeris manager;
        # when requested it must be wired by the caller (kept fail-closed here).
        if third_body:
            raise NotImplementedError(
                "third-body validation requires an ephemeris-wired configuration; "
                "run it through the standard mission config path"
            )
        self._engine = DynamicsEngine(
            sc_props=SpacecraftProps(mass_kg=100.0, area_m2=1.0, cr=1.3),
            flags=PerturbationFlags(enable_sh=True),
            gravity_model=model,
            ephem_manager=None,
            surface_provider=None,
            earth_j2=None,
            allow_identity_rotation=True,
        )
        self._rtol = float(rtol)
        self._atol = float(atol)
        self._method = str(method)
        self.backend_label = f"classical_sh_deg{int(degree)}"
        self.provenance = {
            **_gravity_provenance(model, degree, gravity_file),
            "backend": self.backend_label,
            "integrator": self._method,
            "rtol": self._rtol,
            "atol": self._atol,
        }

    def propagate(
        self, y0: np.ndarray, duration_s: float, output_dt_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        from lunaris.common.type_defs import PropagatorConfig, TimeConfig
        from lunaris.core.propagation.propagator import propagate

        cfg = PropagatorConfig(method=self._method, rtol=self._rtol, atol=self._atol)
        time_cfg = TimeConfig(
            duration_s=float(duration_s), output_dt_s=float(output_dt_s), t0_s=0.0
        )
        result = propagate(
            self._engine, np.asarray(y0, dtype=np.float64), cfg, time_cfg=time_cfg
        )
        t = np.asarray(result.t, dtype=np.float64)
        y = np.asarray(result.y, dtype=np.float64)  # (T, n_state) row-major
        return t, y[:, :6]


__all__ = [
    "ClassicalSHValidationPropagator",
    "TorchSHScreeningPropagator",
]
