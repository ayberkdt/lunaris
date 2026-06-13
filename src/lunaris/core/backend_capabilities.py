# -*- coding: utf-8 -*-
"""
Central Backend Capability Registry
===================================

Single source of truth (SSOT) for *what each propagator backend can do*.

Phase 2 §4 requires CLI, UI, the Monte Carlo engine, the benchmark runner, and
the report/provenance writers to consult **one** capability source so they all
make the same backend decision and label results identically. Before this
module, the same facts were declared in two places:

* :func:`lunaris.core.mc_propagator.gpu_unsupported_features` — classic-SH CUDA
  kernel: blocks albedo, thermal IR, and solid tides.
* :func:`lunaris.core.mc_backend_policy._st_lrps_gpu_unsupported_features` —
  ST-LRPS torch CUDA path: gravity only.

This registry consolidates those facts. It **does not change behavior**: the
values here are locked to the existing MC path by the consistency tests in
``tests/test_backend_capabilities.py``. The classic-SH GPU degree limit is still
sourced from the real workspace constant (:data:`mc_propagator.GPU_SH_MAX_DEGREE`)
via :func:`gpu_sh_max_degree`; the static ``max_sh_degree`` recorded here is
guarded against drift by a test.

Backend names match the ``actual_backend`` strings emitted by
:class:`lunaris.core.mc_backend_policy.MCBackendPlan` and the request names in
``MC_BACKEND_REQUESTS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

# Mapping of canonical force-model name -> the PerturbationFlags attribute that
# enables it. Used to translate an active-flags object into the set of force
# models a backend cannot honor. Keys are the canonical labels surfaced to users
# and written into artifact metadata.
FORCE_MODEL_FLAG_ATTR: dict[str, str] = {
    "spherical_harmonics": "enable_sh",
    "third_body_sun": "enable_3rd_body_sun",
    "third_body_earth": "enable_3rd_body_earth",
    "earth_j2": "enable_earth_j2",
    "srp": "enable_srp",
    "albedo": "enable_albedo",
    "thermal_ir": "enable_thermal",
    "solid_tides_k2": "enable_tides_k2",
    "solid_tides_k3": "enable_tides_k3",
    "relativity_1pn": "enable_relativity_1pn",
}


@dataclass(frozen=True)
class BackendCapabilities:
    """Immutable description of one propagator backend's capabilities.

    ``max_sh_degree`` is the highest classic spherical-harmonics degree the
    backend can evaluate on-device. ``None`` means "not bounded by the backend"
    (the CPU full-fidelity path, bounded only by the loaded gravity model) or
    "not applicable" (the ST-LRPS surrogate, which is not an SH evaluator).

    The ``supports_*`` flags map to :class:`~lunaris.common.type_defs.PerturbationFlags`.
    ``supports_third_body`` covers both Sun and Earth third-body terms;
    ``supports_solid_tides`` covers both the k2 and k3 contributions — no current
    backend supports one half of either pair without the other.
    """

    name: str
    device: str                 # "cpu" | "cuda" | "auto"
    gravity_kind: str           # "classic_sh" | "st_lrps" | "any"
    max_sh_degree: Optional[int]
    supports_float32: bool
    supports_float64: bool
    supports_sh: bool
    supports_third_body: bool
    supports_earth_j2: bool
    supports_srp: bool
    supports_albedo: bool
    supports_thermal: bool
    supports_solid_tides: bool
    supports_relativity_1pn: bool
    integrator: str
    default_dtype: str
    is_meta: bool = False
    description: str = ""

    def supports_force_model(self, canonical_name: str) -> bool:
        """Return whether this backend can model the named canonical force model."""
        mapping = {
            "spherical_harmonics": self.supports_sh,
            "third_body_sun": self.supports_third_body,
            "third_body_earth": self.supports_third_body,
            "earth_j2": self.supports_earth_j2,
            "srp": self.supports_srp,
            "albedo": self.supports_albedo,
            "thermal_ir": self.supports_thermal,
            "solid_tides_k2": self.supports_solid_tides,
            "solid_tides_k3": self.supports_solid_tides,
            "relativity_1pn": self.supports_relativity_1pn,
        }
        if canonical_name not in mapping:
            raise KeyError(f"Unknown force model {canonical_name!r}")
        return bool(mapping[canonical_name])


# =============================================================================
# Registry
# =============================================================================
#
# Capability values below are FAITHFUL to the current implementation, not the
# illustrative example in the Phase 2 brief:
#   * classic-SH GPU and CPU run float64 (no float32 kernel today);
#   * the ST-LRPS torch path defaults to float32 but can run float64.
# The force-model matrix is locked to the existing MC behavior by the
# consistency tests; do not change it without updating those tests.

_CPU_SH = BackendCapabilities(
    name="cpu_sh",
    device="cpu",
    gravity_kind="classic_sh",
    max_sh_degree=None,            # bounded only by the loaded gravity model
    supports_float32=False,
    supports_float64=True,
    supports_sh=True,
    supports_third_body=True,
    supports_earth_j2=True,
    supports_srp=True,
    supports_albedo=True,
    supports_thermal=True,
    supports_solid_tides=True,
    supports_relativity_1pn=True,
    integrator="adaptive (DOP853)",
    default_dtype="float64",
    description="CPU full-fidelity per-sample scipy DOP853. All force models supported.",
)

_GPU_SH = BackendCapabilities(
    name="gpu_sh",
    device="cuda",
    gravity_kind="classic_sh",
    max_sh_degree=24,             # == mc_propagator.GPU_SH_MAX_DEGREE (guarded by test)
    supports_float32=False,
    supports_float64=True,
    supports_sh=True,
    supports_third_body=True,
    supports_earth_j2=True,
    supports_srp=True,
    supports_albedo=False,
    supports_thermal=False,
    supports_solid_tides=False,
    supports_relativity_1pn=True,
    integrator="fixed-step RK4",
    default_dtype="float64",
    description=(
        "Numba CUDA classic-SH fixed-step RK4 (degree <= 24). Supports third-body "
        "Sun/Earth, Earth J2, SRP, and 1PN relativity; albedo, thermal IR, and "
        "solid tides require the CPU backend."
    ),
)

_GPU_ST_LRPS_POTENTIAL = BackendCapabilities(
    name="gpu_st_lrps_potential",
    device="cuda",
    gravity_kind="st_lrps",
    max_sh_degree=None,           # surrogate gravity, not an SH evaluator
    supports_float32=True,
    supports_float64=True,
    supports_sh=True,             # provides central gravity via the surrogate
    supports_third_body=False,
    supports_earth_j2=False,
    supports_srp=False,
    supports_albedo=False,
    supports_thermal=False,
    supports_solid_tides=False,
    supports_relativity_1pn=False,
    integrator="fixed-step RK4",
    default_dtype="float32",
    description=(
        "PyTorch CUDA fixed-step RK4; ST-LRPS acceleration via batched autograd. "
        "Gravity only — any added perturbation forces a CPU fallback."
    ),
)

_GPU_ST_LRPS_DIRECT = BackendCapabilities(
    name="gpu_st_lrps_direct",
    device="cuda",
    gravity_kind="st_lrps",
    max_sh_degree=None,
    supports_float32=True,
    supports_float64=True,
    supports_sh=True,
    supports_third_body=False,
    supports_earth_j2=False,
    supports_srp=False,
    supports_albedo=False,
    supports_thermal=False,
    supports_solid_tides=False,
    supports_relativity_1pn=False,
    integrator="fixed-step RK4",
    default_dtype="float32",
    description=(
        "PyTorch CUDA fixed-step RK4; ST-LRPS direct residual acceleration via a "
        "batched no-grad forward pass. Gravity only."
    ),
)

# CPU ST-LRPS full-fidelity path (the actual_backend the policy emits when an
# ST-LRPS GPU run is forced back to CPU). Not in the brief's five-name list but
# emitted by the system, so it is registered for complete provenance labeling.
_CPU_ST_LRPS = BackendCapabilities(
    name="cpu_st_lrps",
    device="cpu",
    gravity_kind="st_lrps",
    max_sh_degree=None,
    supports_float32=False,
    supports_float64=True,
    supports_sh=True,
    supports_third_body=True,
    supports_earth_j2=True,
    supports_srp=True,
    supports_albedo=True,
    supports_thermal=True,
    supports_solid_tides=True,
    supports_relativity_1pn=True,
    integrator="adaptive (DOP853)",
    default_dtype="float64",
    description="CPU full-fidelity DOP853 with the ST-LRPS surrogate as the gravity model.",
)

# Meta request: the policy resolver picks the concrete backend at runtime.
_AUTO = BackendCapabilities(
    name="auto",
    device="auto",
    gravity_kind="any",
    max_sh_degree=None,
    supports_float32=True,
    supports_float64=True,
    supports_sh=True,
    supports_third_body=True,
    supports_earth_j2=True,
    supports_srp=True,
    supports_albedo=True,
    supports_thermal=True,
    supports_solid_tides=True,
    supports_relativity_1pn=True,
    integrator="resolved at runtime",
    default_dtype="resolved at runtime",
    is_meta=True,
    description="Meta request resolved to a concrete backend by mc_backend_policy.",
)

BACKEND_REGISTRY: dict[str, BackendCapabilities] = {
    cap.name: cap
    for cap in (
        _CPU_SH,
        _GPU_SH,
        _GPU_ST_LRPS_POTENTIAL,
        _GPU_ST_LRPS_DIRECT,
        _CPU_ST_LRPS,
        _AUTO,
    )
}

# The five request/backend names the Phase 2 brief requires to be defined.
REQUIRED_BACKEND_NAMES: Tuple[str, ...] = (
    "cpu_sh",
    "gpu_sh",
    "gpu_st_lrps_potential",
    "gpu_st_lrps_direct",
    "auto",
)


# =============================================================================
# Public helpers
# =============================================================================


def get_capabilities(name: str) -> BackendCapabilities:
    """Return the :class:`BackendCapabilities` for ``name``.

    Raises :class:`KeyError` with the list of known names if ``name`` is not a
    registered backend (callers should surface this as a configuration error
    *before* a run starts, never silently fall back).
    """
    try:
        return BACKEND_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(BACKEND_REGISTRY))
        raise KeyError(f"Unknown backend {name!r}; known backends: {known}") from None


def list_backend_names() -> Tuple[str, ...]:
    """Return all registered backend names in a stable, sorted order."""
    return tuple(sorted(BACKEND_REGISTRY))


def _flag_on(flags: Any, attr: str) -> bool:
    return bool(getattr(flags, attr, False))


def unsupported_force_models(name: str, flags: Any) -> Tuple[str, ...]:
    """Return the canonical force models that are *active* in ``flags`` but not
    supported by backend ``name``.

    An empty tuple means the backend can honor every active force model. A
    non-empty result means a run on this backend would silently drop physics —
    callers must treat it as a hard error or an explicit, recorded fallback,
    never a silent one.
    """
    caps = get_capabilities(name)
    blocked: list[str] = []
    for canonical, attr in FORCE_MODEL_FLAG_ATTR.items():
        if canonical == "spherical_harmonics":
            continue  # central gravity, not an optional perturbation
        if _flag_on(flags, attr) and not caps.supports_force_model(canonical):
            blocked.append(canonical)
    return tuple(blocked)


def gpu_sh_max_degree() -> int:
    """Return the true classic-SH GPU degree limit from the CUDA kernel workspace.

    Sourced from :data:`lunaris.core.mc_propagator.GPU_SH_MAX_DEGREE` (lazily, to
    avoid importing the Numba CUDA stack at module load). Falls back to the
    historical default of 24 if that import is unavailable.
    """
    try:
        from lunaris.core.mc_propagator import GPU_SH_MAX_DEGREE

        return int(GPU_SH_MAX_DEGREE)
    except Exception:
        return 24


def gpu_sh_supported_tiers() -> Tuple[int, ...]:
    """Return the supported true-GPU classic-SH degree tiers."""
    try:
        from lunaris.core.mc_propagator import GPU_SH_SUPPORTED_TIERS

        return tuple(int(v) for v in GPU_SH_SUPPORTED_TIERS)
    except Exception:
        return (gpu_sh_max_degree(),)


__all__ = [
    "BackendCapabilities",
    "BACKEND_REGISTRY",
    "REQUIRED_BACKEND_NAMES",
    "FORCE_MODEL_FLAG_ATTR",
    "get_capabilities",
    "list_backend_names",
    "unsupported_force_models",
    "gpu_sh_max_degree",
    "gpu_sh_supported_tiers",
]
