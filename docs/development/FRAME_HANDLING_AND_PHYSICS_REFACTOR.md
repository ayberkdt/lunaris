# Frame Handling and Physics Refactor Notes

Status: development note for the frame-handling and physics refactor session.
The runtime behavior is unchanged by this document.

This note groups the follow-up items that should stay discoverable for future
work without spreading short planning files across the root of `docs/`.

## GPU Smoke Tests

These checks are backend-initialization and short-propagation smokes only. They
are not paper-safe benchmarks and must not be reported as throughput evidence.

Normal CI remains CPU-only. CUDA-dependent tests should use the existing
`@pytest.mark.requires_cuda` marker and skip cleanly when no NVIDIA CUDA device
is available.

Use a developer machine with:

- the Lunaris optional GPU stack installed;
- a functional PyTorch install for `torch_cuda_sh` and GPU ST-LRPS paths;
- a CUDA-capable NVIDIA device visible to the Python process;
- local data/artifacts needed by the selected backend.

Record requested and actual backend provenance after every run. A CPU fallback is
a valid smoke result only when the policy explicitly records it as fallback; it
is not a GPU smoke pass.

Minimal commands:

```bash
lunaris-batch --batch-backend torch_cpu_sh --n-samples 8 --days 0.01
lunaris-batch --batch-backend torch_cuda_sh --n-samples 8 --days 0.01
lunaris-batch --batch-backend gpu_st_lrps_potential --n-samples 8 --days 0.01
lunaris-batch --batch-backend gpu_st_lrps_third_body --n-samples 8 --days 0.01
```

Run CUDA-marked tests explicitly on a CUDA host:

```bash
python -m pytest -m requires_cuda
```

For CPU-only CI or laptops without CUDA:

```bash
python -m pytest -m "not requires_cuda"
```

The marker is registered in `pyproject.toml`. Tests that require both real data
and CUDA should be marked with both `requires_data` and `requires_cuda`.

Acceptance checklist:

- `requested_batch_backend` matches the command.
- `actual_batch_backend` is the requested GPU backend for GPU smoke passes.
- `actual_device` names a CUDA device for CUDA smoke passes.
- `fallback_applied` is false for a claimed GPU pass.
- `unsupported_forces` is empty, except for tests intentionally exercising
  fallback.
- Output archive metadata includes dtype, integrator, requested/actual SH
  degree, and backend implementation.

If any physics flag is unsupported by the requested GPU path, the correct result
is an explicit recorded fallback or a hard error under strict policy, not a
silent GPU run with the force dropped.

## Optional Force Probe

Single-run debugging would benefit from an opt-in diagnostic payload such as:

```python
diagnostics["force_probe"] = {
    "t0": {
        "central_gravity": ...,
        "third_body_sun": ...,
        "third_body_earth": ...,
        "earth_j2": ...,
        "srp": ...,
        "albedo": ...,
        "thermal_ir": ...,
        "solid_tides": ...,
        "relativity": ...,
        "total": ...,
    }
}
```

The probe must be disabled by default and must not add work to normal
propagation runs.

`DynamicsEngine.get_acceleration_breakdown(t, y)` already exposes component
norms for reporting and debugging. It rebuilds the same prepared packs used by
the RHS and evaluates contributions in SI units. The
`tests/test_force_budget_sanity.py` regression suite uses that API to lock down
finite, nonzero, order-of-magnitude force budgets at 100 km, 300 km, and
1000 km lunar altitudes without changing the runtime RHS.

The RHS closures still own most contribution accumulation internally. A fully
structured `force_probe` should return vectors, frames, and totals, not only
norms. Adding that inside `propagate()` now would either duplicate force logic
or thread debug callbacks through hot-loop code that is deliberately allocation
light.

If implemented later, add a `PropagatorConfig.enable_force_probe` flag and keep
the following constraints:

- off by default;
- sample only a small fixed set of epochs such as initial/final state;
- report SI units and inertial-vs-fixed frame explicitly;
- use existing prepared force packs instead of reloading models;
- never disable or simplify a force to make probing cheaper;
- include `force_probe_schema_version`.

The design should reuse `get_acceleration_breakdown()` or a vector-valued
sibling owned by `lunaris.core.dynamics`; `lunaris.core.propagation` should not
duplicate force-model math.

## Planetary Third-Body Generalization

Status: P3 design note. Not implemented in this refactor session.

The current production force model intentionally focuses on Sun and Earth as the
dominant external bodies for lunar orbit propagation. Those paths should remain
fast and explicit. A future extension may generalize the external-body plumbing
without slowing today's hot loop.

Proposed contract:

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class ExternalBodyPerturber:
    name: str
    mu_m3s2: float
    position_table_m: np.ndarray
    velocity_table_m_s: np.ndarray | None
    enabled_for_point_mass: bool
    enabled_for_relativity: bool
    enabled_for_tides: bool
```

The table frame must be Moon-centered inertial unless a caller explicitly
documents another frame and provides a rotation bridge. Units are SI.

Intended uses:

- Add Jupiter, Venus, or other planetary point-mass perturbations for
  long-duration high-accuracy comparisons.
- Share one external-body table contract for third-body point mass, external
  1PN relativity, and tide-raising bodies.
- Preserve the current Sun/Earth fast path for normal runtime.

Non-goals:

- No spacecraft attitude or geometry modeling.
- No change to the current Sun/Earth numerical formulas.
- No hot-loop Python iteration over arbitrary bodies in the default path.
- No new ephemeris dependency at import time.

Implementation sketch:

1. Keep `r_sun_tab_m`, `r_earth_tab_m`, and `q_i2f_tab` as the default prepared
   pack fields so current Numba closures remain unchanged.
2. Add an optional external-body pack for non-default bodies. It should be empty
   unless explicitly requested by config.
3. Compile specialized kernels for the active body count or pre-sum the small
   body list in setup where possible. Do not add generic object loops inside
   Numba hot paths.
4. Record provenance for each body: name, gravitational parameter, ephemeris
   source, table cadence, and whether velocity is present.
5. Extend backend capability metadata before enabling a body on GPU. A backend
   that cannot model a requested body must fallback or error explicitly.

Validation requirements:

- Differential third-body acceleration must match the existing Sun/Earth kernel
  for Sun/Earth when expressed through the generic contract.
- External 1PN terms must continue to use body velocity tables when enabled.
- Solid-tide frame transforms must be checked against the existing Moon-fixed
  implementation.
- CPU reference tests should cover at least one additional body with broad
  order-of-magnitude and sign/limiting-case checks.
- GPU paths need matched CPU/GPU agreement before any performance claim.

This note is deliberately a design target, not a partial implementation. The
current refactor keeps the runtime behavior unchanged.
