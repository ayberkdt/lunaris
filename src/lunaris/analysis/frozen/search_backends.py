"""Concrete propagation backends for the frozen-orbit search pipeline.

These adapters wire the injected pipeline protocols to real Lunaris paths:

- :class:`TorchSHScreeningPropagator` — batch screening on the shared R07
  fixed-step loop via :class:`~lunaris.core.torch_sh_propagator.TorchSHBatchPropagator`
  (classical SH physics, CPU or CUDA; honest backend naming, VRAM-aware
  chunking, recorded provenance).
- :class:`STLRPSScreeningPropagator` - CUDA ST-LRPS screening on
  :class:`~lunaris.core.torch_batch_propagator.TorchBatchPropagator`, with the
  supported ``potential_autograd`` runtime and optional analytic Sun/Earth
  third-body terms.
- :class:`ClassicalSHValidationPropagator` — single-orbit CPU reference
  propagation (adaptive DOP853 through :func:`lunaris.core.propagation.propagator.propagate`),
  the only backend whose label satisfies the R21 classical-SH rule.

Both keep torch/Numba imports lazy so ``lunaris.analysis.frozen`` stays
importable without the optional heavy dependencies.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
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


def normalize_third_body_selection(value: Any) -> tuple[str, ...]:
    """Normalize CLI/API third-body selectors to ``("sun", "earth")`` tokens."""
    if value is None:
        return ()
    if isinstance(value, bool):
        return ("sun", "earth") if value else ()
    if isinstance(value, str):
        text = value.strip().lower().replace("+", ",").replace(";", ",")
        if text in {"", "0", "false", "no", "none", "off"}:
            return ()
        if text in {"both", "all", "sun_earth", "earth_sun"}:
            tokens = ["sun", "earth"]
        else:
            tokens = [part.strip() for part in text.split(",") if part.strip()]
    else:
        tokens = [str(part).strip().lower() for part in value if str(part).strip()]

    selected: set[str] = set()
    for token in tokens:
        cleaned = token.replace("-", "_")
        if cleaned in {"sun", "third_body_sun", "3b_sun"}:
            selected.add("sun")
        elif cleaned in {"earth", "third_body_earth", "3b_earth"}:
            selected.add("earth")
        elif cleaned in {"both", "all"}:
            selected.update(("sun", "earth"))
        else:
            raise ValueError(
                "third-body selector must be 'none', 'sun', 'earth', or 'sun,earth'; "
                f"got {token!r}"
            )
    return tuple(body for body in ("sun", "earth") if body in selected)


def _torch_third_body_selectors(bodies: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"third_body_{body}" for body in bodies)


def _third_body_provenance(bodies: tuple[str, ...]) -> dict[str, bool]:
    selected = set(bodies)
    return {"earth": "earth" in selected, "sun": "sun" in selected}


def _frame_provenance(ephem_manager: Any, *, allow_identity_rotation: bool) -> str:
    if ephem_manager is not None:
        return "moon_fixed_slerp (ephemeris-wired q_i2f)"
    if allow_identity_rotation:
        return "identity (gravity field fixed in the integration frame)"
    return "unresolved (ephemeris required)"


def _gravity_provenance(
    model: Any,
    degree: int,
    gravity_file: str | None,
    *,
    third_body: tuple[str, ...] = (),
    ephem_manager: Any = None,
    allow_identity_rotation: bool = True,
) -> dict[str, Any]:
    return {
        "gravity_model": {
            "name": str(getattr(model, "name", "unknown")),
            "file": str(gravity_file) if gravity_file else "default_config",
            "degree": int(degree),
            "loaded_degree_max": int(getattr(model, "degree_max", 0)),
        },
        "third_body": _third_body_provenance(third_body),
        "force_model_scope": "gravity_plus_third_body" if third_body else "gravity_only",
        "frame": _frame_provenance(
            ephem_manager, allow_identity_rotation=allow_identity_rotation
        ),
    }


def build_ephemeris_manager_for_frozen_search(
    *,
    duration_s: float,
    output_dt_s: float,
    start_date: str | None = None,
    include_third_body: bool = True,
) -> Any:
    """Build the ephemeris table bundle used by frozen-search backends."""
    from lunaris.common.constants import DAY_S
    from lunaris.core.config import load_default_config, replace_sim_config
    from lunaris.physics.ephemeris import EphemerisManager

    cfg = load_default_config()
    time_updates: dict[str, Any] = {
        "duration_s": float(duration_s) + 0.1 * DAY_S,
        "output_dt_s": float(output_dt_s),
    }
    if start_date:
        time_updates["start_date"] = str(start_date)
    cfg = replace_sim_config(
        cfg,
        time=replace(cfg.time, **time_updates),
        spice=replace(cfg.spice, include_third_body=bool(include_third_body)),
    )
    return EphemerisManager.from_time_and_spice(
        cfg.time,
        cfg.spice,
        auto_fix_kernel_paths=True,
        need_moon_fixed_rotation=True,
    )


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


class STLRPSScreeningPropagator:
    """Stage-1 batch screening on the CUDA ST-LRPS propagation backend."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        dt_s: float = 60.0,
        device_id: int = 0,
        torch_dtype: str = "float32",
        chunk_size: int | None = None,
        ephem_manager: Any = None,
        allow_identity_rotation: bool = False,
        third_body: Any = (),
        strict_domain: bool = True,
    ) -> None:
        from lunaris.common.batch_defs import (
            BatchPropagationConfig,
            validate_st_lrps_model_dir,
        )
        from lunaris.core.torch_batch_propagator import TorchBatchPropagator
        from lunaris.surrogate.runtime import SurrogateGravityModel

        bodies = normalize_third_body_selection(third_body)
        if bodies and ephem_manager is None:
            raise RuntimeError(
                "ST-LRPS third-body screening requires an ephemeris manager with "
                "Sun/Earth position tables."
            )

        valid_dir = validate_st_lrps_model_dir(model_dir)
        model = SurrogateGravityModel.from_model_dir(
            str(valid_dir),
            device_preference="cpu",
            strict_domain=bool(strict_domain),
        )
        backend_name = "gpu_st_lrps_third_body" if bodies else "gpu_st_lrps_potential"
        batch_cfg = BatchPropagationConfig(
            n_samples=2,
            seed=0,
            use_gpu=True,
            batch_backend=backend_name,
            gravity_mode_override="st_lrps",
            st_lrps_model_dir=str(valid_dir),
            gpu_device_id=int(device_id),
            dt_s=float(dt_s),
            torch_dtype=str(torch_dtype).lower(),
            torch_sh_chunk_size=int(chunk_size or 0),
            impact_alt_km=0.0,
            output_mode="full",
            summary_top_k=1,
        )
        self._propagator = TorchBatchPropagator(
            surrogate_model=model,
            batch_cfg=batch_cfg,
            device_id=int(device_id),
            ephem=ephem_manager,
            allow_identity_rotation=bool(allow_identity_rotation),
            third_body=_torch_third_body_selectors(bodies),
        )
        runtime_kind = str(
            getattr(getattr(model, "_force_runtime", None), "runtime_model_kind", "")
            or getattr(model, "config", {}).get("runtime_model_kind", "potential_autograd")
        )
        self.backend_name = backend_name
        self.provenance = {
            "backend": self.backend_name,
            "device": f"cuda:{int(device_id)}",
            "dtype": str(torch_dtype).lower(),
            "dt_s": float(dt_s),
            "loop": "TorchBatchPropagator fixed-step RK4",
            "gravity_model": {
                "backend": "st_lrps",
                "model_dir": str(valid_dir),
                "degree_min": int(getattr(model, "degree_min", 0)),
                "degree_max": int(getattr(model, "degree_max", 0)),
                "effective_degree_max": int(getattr(model, "effective_degree_max", 0)),
                "runtime_model_kind": runtime_kind,
            },
            "third_body": _third_body_provenance(bodies),
            "force_model_scope": (
                "st_lrps_plus_third_body" if bodies else "st_lrps_gravity_only"
            ),
            "frame": _frame_provenance(
                ephem_manager,
                allow_identity_rotation=bool(allow_identity_rotation),
            ),
            "strict_domain": bool(strict_domain),
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
        third_body: Any = False,
        ephem_manager: Any = None,
        allow_identity_rotation: bool | None = None,
    ) -> None:
        from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
        from lunaris.core.dynamics import DynamicsEngine

        model = _load_gravity_model(gravity_file, degree)
        bodies = normalize_third_body_selection(third_body)
        if bodies and ephem_manager is None:
            raise RuntimeError(
                "third-body validation requires an ephemeris-wired configuration "
                "(Sun/Earth position tables plus Moon-fixed rotation)."
            )
        if allow_identity_rotation is None:
            allow_identity_rotation = ephem_manager is None
        flags = PerturbationFlags(
            enable_sh=True,
            enable_3rd_body_sun="sun" in bodies,
            enable_3rd_body_earth="earth" in bodies,
        )
        self._engine = DynamicsEngine(
            sc_props=SpacecraftProps(mass_kg=100.0, area_m2=1.0, cr=1.3),
            flags=flags,
            gravity_model=model,
            ephem_manager=ephem_manager,
            surface_provider=None,
            earth_j2=None,
            allow_identity_rotation=bool(allow_identity_rotation),
        )
        self._rtol = float(rtol)
        self._atol = float(atol)
        self._method = str(method)
        suffix = "" if not bodies else "_3b_" + "_".join(bodies)
        self.backend_label = f"classical_sh_deg{int(degree)}{suffix}"
        self.provenance = {
            **_gravity_provenance(
                model,
                degree,
                gravity_file,
                third_body=bodies,
                ephem_manager=ephem_manager,
                allow_identity_rotation=bool(allow_identity_rotation),
            ),
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
    "STLRPSScreeningPropagator",
    "TorchSHScreeningPropagator",
    "build_ephemeris_manager_for_frozen_search",
    "normalize_third_body_selection",
]
