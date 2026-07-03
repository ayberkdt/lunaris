"""
Central Backend Capability Registry
===================================

Single source of truth (SSOT) for *what each propagator backend can do*.

CLI, UI, the batch propagation engine, the benchmark runner, and the
report/provenance writers consult **one** capability source so they all make the
same backend decision and label results identically.

Two distinct GPU spherical-harmonics implementations exist in the repository and
are kept **separate** here — they must never be merged under a single ``gpu_sh``
capability:

* ``numba_cuda_sh`` — Numba CUDA, one ensemble sample per CUDA thread with a
  fixed thread-local Legendre workspace. The degree-24 ceiling is a *kernel
  workspace* limit (``cuda.local.array``), not a physical one. Intended for
  low-degree, high-throughput batch screening.
* ``torch_cuda_sh`` — PyTorch tensors on CUDA (or CPU), evaluating arbitrary
  degrees (SH25/50/100/200…) bounded only by the loaded coefficient file, GPU
  memory, batch size, dtype, and step size — **no** hard degree-24 cap.

``gpu_sh`` is retained only as a **legacy alias** that resolves explicitly to
``numba_cuda_sh`` (see :data:`BACKEND_ALIASES`). Machine-readable provenance and
user-facing output always use the resolved, real backend name.

The Numba force-model support matrix here is locked to the existing batch
behavior by the consistency tests in ``tests/test_backend_capabilities.py``; do
not change it without updating those tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Mapping of canonical force-model name -> the PerturbationFlags attribute that
# enables it. Used to translate an active-flags object into the set of force
# models a backend cannot honor.
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

    ``max_runtime_sh_degree`` is the highest classic spherical-harmonics degree
    the backend can evaluate on-device. ``None`` does **not** mean "infinite
    physical accuracy" or "unlimited GPU capacity"; it means only that the
    *implementation* has no hard-coded SH-degree ceiling (the real limit is the
    coefficient file, GPU memory, batch size, dtype, and step size). A concrete
    integer (e.g. 24 for ``numba_cuda_sh``) is an implementation ceiling that
    must never be silently exceeded by clipping.

    ``family`` groups backends by physics family (``classic_sh`` / ``st_lrps`` /
    ``meta``); ``implementation`` names the compute backend
    (``numba_cuda`` / ``torch`` / ``numpy_numba_cpu`` …). The ``supports_*``
    flags map to :class:`~lunaris.common.type_defs.PerturbationFlags`;
    ``supports_third_body`` covers both Sun and Earth, ``supports_solid_tides``
    covers both k2 and k3.
    """

    name: str
    family: str                       # "classic_sh" | "st_lrps" | "meta"
    implementation: str               # "numba_cuda" | "torch" | "numpy_numba_cpu" | ...
    device: str                       # "cpu" | "cuda" | "auto"
    max_runtime_sh_degree: int | None
    dtype_support: tuple[str, ...]
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
    fidelity_class: str = "standard"
    is_meta: bool = False
    description: str = ""

    # ---- Back-compat aliases (older callers used these names) -------------
    @property
    def gravity_kind(self) -> str:
        """Alias of :attr:`family` kept for backward compatibility."""
        return self.family

    @property
    def max_sh_degree(self) -> int | None:
        """Alias of :attr:`max_runtime_sh_degree` kept for backward compatibility."""
        return self.max_runtime_sh_degree

    @property
    def supports_float32(self) -> bool:
        return "float32" in self.dtype_support

    @property
    def supports_float64(self) -> bool:
        return "float64" in self.dtype_support

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
# Values are FAITHFUL to the current implementation, not the illustrative brief:
#   * classic-SH Numba GPU and CPU run float64 (no float32 SH kernel today);
#   * the torch SH path can run float32 or float64;
#   * the torch SH batch runtime is, in its first form, gravity-only — any
#     extra perturbation forces an explicit fallback (see §7 of the task brief).

_CPU_SH = BackendCapabilities(
    name="cpu_sh",
    family="classic_sh",
    implementation="numpy_numba_cpu",
    device="cpu",
    max_runtime_sh_degree=None,        # bounded only by the loaded gravity model
    dtype_support=("float64",),
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
    fidelity_class="full",
    description="CPU full-fidelity per-sample scipy DOP853. All force models supported.",
)

_NUMBA_CUDA_SH = BackendCapabilities(
    name="numba_cuda_sh",
    family="classic_sh",
    implementation="numba_cuda",
    device="cuda",
    max_runtime_sh_degree=24,          # == batch_propagator.GPU_SH_MAX_DEGREE (guarded by test)
    dtype_support=("float64",),
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
    fidelity_class="low_degree_screening",
    description=(
        "Numba CUDA classic-SH fixed-step RK4, one batch sample per CUDA thread "
        "(degree <= 24, a thread-local workspace limit — NOT a physical one). "
        "Supports third-body Sun/Earth, Earth J2, SRP, and 1PN relativity; "
        "albedo, thermal IR, and solid tides require the CPU backend. Use for "
        "low-degree, high-throughput batch screening."
    ),
)

# Torch classic-SH backends. First runtime form is gravity-only (lunar SH +
# frame transform + fixed-step batch propagation); added perturbations trigger
# an explicit, recorded fallback rather than silent disabling.
_TORCH_CUDA_SH = BackendCapabilities(
    name="torch_cuda_sh",
    family="classic_sh",
    implementation="torch",
    device="cuda",
    max_runtime_sh_degree=None,        # no hard degree cap; limited by coeff file / VRAM / batch
    dtype_support=("float32", "float64"),
    supports_sh=True,
    supports_third_body=False,
    supports_earth_j2=False,
    supports_srp=False,
    supports_albedo=False,
    supports_thermal=False,
    supports_solid_tides=False,
    supports_relativity_1pn=False,
    integrator="fixed-step RK4",
    default_dtype="float64",
    fidelity_class="high_degree",
    description=(
        "PyTorch CUDA classic-SH, arbitrary degree (SH25/50/100/200…) bounded by "
        "the coefficient file, GPU memory, batch size, dtype, and step size. "
        "First runtime form is gravity-only."
    ),
)

_TORCH_CPU_SH = BackendCapabilities(
    name="torch_cpu_sh",
    family="classic_sh",
    implementation="torch",
    device="cpu",
    max_runtime_sh_degree=None,
    dtype_support=("float32", "float64"),
    supports_sh=True,
    supports_third_body=False,
    supports_earth_j2=False,
    supports_srp=False,
    supports_albedo=False,
    supports_thermal=False,
    supports_solid_tides=False,
    supports_relativity_1pn=False,
    integrator="fixed-step RK4",
    default_dtype="float64",
    fidelity_class="high_degree",
    description=(
        "PyTorch CPU classic-SH (same evaluator as torch_cuda_sh on CPU). Useful "
        "for validation and CUDA-free testing of the high-degree SH path."
    ),
)

_GPU_ST_LRPS_POTENTIAL = BackendCapabilities(
    name="gpu_st_lrps_potential",
    family="st_lrps",
    implementation="torch",
    device="cuda",
    max_runtime_sh_degree=None,        # surrogate gravity, not an SH evaluator
    dtype_support=("float32", "float64"),
    supports_sh=True,                  # provides central gravity via the surrogate
    supports_third_body=False,
    supports_earth_j2=False,
    supports_srp=False,
    supports_albedo=False,
    supports_thermal=False,
    supports_solid_tides=False,
    supports_relativity_1pn=False,
    integrator="fixed-step RK4",
    default_dtype="float32",
    fidelity_class="surrogate",
    description=(
        "PyTorch CUDA fixed-step RK4; ST-LRPS = torch SH baseline + neural "
        "residual via batched autograd. Gravity only — any added perturbation "
        "forces a CPU fallback."
    ),
)

_GPU_ST_LRPS_DIRECT = BackendCapabilities(
    name="gpu_st_lrps_direct",
    family="st_lrps",
    implementation="torch",
    device="cuda",
    max_runtime_sh_degree=None,
    dtype_support=("float32", "float64"),
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
    fidelity_class="surrogate",
    description=(
        "PyTorch CUDA fixed-step RK4; ST-LRPS direct residual acceleration via a "
        "batched no-grad forward pass. Gravity only."
    ),
)

# CPU ST-LRPS full-fidelity path (the actual_backend emitted when an ST-LRPS GPU
# run is forced back to CPU).
_CPU_ST_LRPS = BackendCapabilities(
    name="cpu_st_lrps",
    family="st_lrps",
    implementation="torch_cpu",
    device="cpu",
    max_runtime_sh_degree=None,
    dtype_support=("float64",),
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
    fidelity_class="full",
    description="CPU full-fidelity DOP853 with the ST-LRPS surrogate as the gravity model.",
)

# Meta request: the policy resolver picks the concrete backend at runtime.
_AUTO = BackendCapabilities(
    name="auto",
    family="meta",
    implementation="resolved",
    device="auto",
    max_runtime_sh_degree=None,
    dtype_support=("float32", "float64"),
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
    description="Meta request resolved to a concrete backend by backend_policy.",
)

BACKEND_REGISTRY: dict[str, BackendCapabilities] = {
    cap.name: cap
    for cap in (
        _CPU_SH,
        _NUMBA_CUDA_SH,
        _TORCH_CUDA_SH,
        _TORCH_CPU_SH,
        _GPU_ST_LRPS_POTENTIAL,
        _GPU_ST_LRPS_DIRECT,
        _CPU_ST_LRPS,
        _AUTO,
    )
}

# Legacy names kept for backward compatibility, resolved to the real backend.
# ``gpu_sh`` was the single classic-SH GPU capability before the Numba/torch
# split; it now resolves explicitly to the Numba CUDA backend.
BACKEND_ALIASES: dict[str, str] = {
    "gpu_sh": "numba_cuda_sh",
}

# Backend names the task brief requires to be registered as distinct entries.
REQUIRED_BACKEND_NAMES: tuple[str, ...] = (
    "cpu_sh",
    "numba_cuda_sh",
    "torch_cuda_sh",
    "torch_cpu_sh",
    "gpu_st_lrps_potential",
    "gpu_st_lrps_direct",
    "cpu_st_lrps",
    "auto",
)


# =============================================================================
# Public helpers
# =============================================================================


def resolve_backend_alias(name: str) -> str:
    """Resolve a (possibly legacy) backend name to its canonical registry name.

    ``gpu_sh`` -> ``numba_cuda_sh``. Non-alias names are returned unchanged.
    """
    key = str(name).strip()
    return BACKEND_ALIASES.get(key, key)


def get_capabilities(name: str) -> BackendCapabilities:
    """Return the :class:`BackendCapabilities` for ``name`` (resolving aliases).

    Raises :class:`KeyError` listing the known names if ``name`` is neither a
    registered backend nor a known alias (callers should surface this as a
    configuration error before a run starts, never a silent fallback).
    """
    resolved = resolve_backend_alias(name)
    try:
        return BACKEND_REGISTRY[resolved]
    except KeyError:
        known = ", ".join(sorted(BACKEND_REGISTRY))
        aliases = ", ".join(sorted(BACKEND_ALIASES))
        raise KeyError(
            f"Unknown backend {name!r}; known backends: {known}; aliases: {aliases}"
        ) from None


def list_backend_names(*, include_aliases: bool = False) -> tuple[str, ...]:
    """Return all registered backend names in a stable, sorted order."""
    names = set(BACKEND_REGISTRY)
    if include_aliases:
        names |= set(BACKEND_ALIASES)
    return tuple(sorted(names))


def _flag_on(flags: Any, attr: str) -> bool:
    return bool(getattr(flags, attr, False))


def unsupported_force_models(name: str, flags: Any) -> tuple[str, ...]:
    """Return the canonical force models active in ``flags`` but unsupported by
    backend ``name``.

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
    """Return the Numba CUDA classic-SH degree limit from the kernel workspace.

    Sourced from :data:`lunaris.core.batch_propagator.GPU_SH_MAX_DEGREE` (lazily, to
    avoid importing the Numba CUDA stack at module load). Falls back to the
    historical default of 24 if that import is unavailable. This is the limit of
    the ``numba_cuda_sh`` backend only; ``torch_cuda_sh`` has no such cap.
    """
    try:
        from lunaris.core.batch_propagator import GPU_SH_MAX_DEGREE

        return int(GPU_SH_MAX_DEGREE)
    except Exception:
        return 24


def gpu_sh_supported_tiers() -> tuple[int, ...]:
    """Return the supported Numba CUDA classic-SH degree tiers."""
    try:
        from lunaris.core.batch_propagator import GPU_SH_SUPPORTED_TIERS

        return tuple(int(v) for v in GPU_SH_SUPPORTED_TIERS)
    except Exception:
        return (gpu_sh_max_degree(),)


__all__ = [
    "BackendCapabilities",
    "BACKEND_REGISTRY",
    "BACKEND_ALIASES",
    "REQUIRED_BACKEND_NAMES",
    "FORCE_MODEL_FLAG_ATTR",
    "resolve_backend_alias",
    "get_capabilities",
    "list_backend_names",
    "unsupported_force_models",
    "gpu_sh_max_degree",
    "gpu_sh_supported_tiers",
]
