---
name: lunaris-architecture-guardian
description: >-
  Enforce the Lunaris layered architecture and single-source-of-truth config
  flow when adding, moving, or refactoring code under src/lunaris/. Use whenever
  a change touches module placement, cross-layer imports, SimConfig/constants,
  optional-dependency boundaries (torch, PySide6, spiceypy, h5py), public CLI
  entry points, or where generated artifacts are written. Trigger on phrases
  like "where should this live", "is this import allowed", "add a new module",
  "refactor", "circular import", or "is this the right layer". NOT for pure
  physics-correctness questions (use astrodynamics-validation), backend/GPU
  routing (use mc-backend-and-performance), or UI styling (use lunaris-ux-design).
---

# Lunaris Architecture Guardian

Lunaris is a four-layer package under `src/lunaris/`. A layer must never import
from a layer above it. Most "where does this go" and "why is there a circular
import" problems are layer violations. This skill keeps placement and the config
single-source-of-truth (SSOT) intact.

## Invocation

Auto-trigger; inline (no isolated context). Read-only analysis plus targeted
edits. It does not run long workflows.

## The layer contract (canonical: `docs/ARCHITECTURE.md`, `AGENTS.md`)

1. **`lunaris.common`** — dependency-light shared layer. `constants.py` is the
   SSOT for physical constants; `type_defs.py` / `montecarlo_defs.py` hold the
   frozen configuration dataclasses. Imports nothing from `physics`/`core`/`ui`.
2. **`lunaris.physics`** — Numba-JIT force-model kernels (spherical_harmonics,
   third-body, SRP, surface, relativity), `ephemeris.py` (SPICE via spiceypy),
   `torch_spherical_harmonics.py`, and surrogate-gravity adapters. **Never
   imports from `core/`.**
3. **`lunaris.core`** — numerical engine: `config.py` (`SimConfig` SSOT),
   `dynamics.py`, `propagator.py`, `events.py`, `mc_backend_policy.py`,
   `monte_carlo_engine.py`, `mc_propagator.py`, `torch_sh_propagator.py`,
   `backend_capabilities.py`.
4. **`lunaris.analysis` / `lunaris.visualization` / `lunaris.ui`** —
   post-processing, reporting, MC analysis, standalone visualization, and the
   PySide6 desktop UI. The ST-LRPS pipeline lives under
   `lunaris.surrogate.st_lrps`.

## Required repository sources (read before prescribing)

- `AGENTS.md` and `docs/ARCHITECTURE.md` — the layer rules and data flow.
- `docs/CONFIG_AND_ARTIFACT_CONTRACTS.md` — config/artifact contract surfaces.
- The actual file(s) you intend to add to or move, plus their imports.

## Procedure

1. **Locate the real layer** of every module in the change. Resolve imports;
   confirm direction (lower may not import higher).
2. **Config flow check.** Configuration must flow through the frozen `SimConfig`
   (`lunaris.core.config`): `load_default_config()` →
   `apply_args_to_config()` (in `lunaris.cli.main`) → `cfg.validate()`. Reject
   ad-hoc kwargs threaded around `SimConfig`. New physical constants belong in
   `lunaris.common.constants`, not redefined locally.
3. **Optional-dependency boundary.** `torch`, `h5py`, `PySide6`/`PyQt6`,
   `pyqtgraph`, `reportlab` are optional extras (`pyproject.toml`
   `[project.optional-dependencies]`: `ml`, `hpc`, `ui`, `reports`). Importing
   them at module top-level in `common`/`physics`/`core` breaks core installs —
   import them lazily inside the function that needs them, behind a clear error.
4. **Adapter duplication.** Before adding a gravity/ephemeris adapter, search for
   an existing one (e.g. `physics/gravity_adapter.py`). Do not add a second
   adapter that forwards the same attributes.
5. **Public-surface check.** The 14 console entry points in `pyproject.toml`
   `[project.scripts]` (`lunaris`, `lunaris-ui`, `lunaris-mc`, `lunaris-train`,
   `lunaris-eval`, `lunaris-benchmark`, …) and artifact contracts are stable
   APIs. Don't rename or change their signatures unless the task asks.
6. **Generated outputs.** Artifacts go under `outputs/` (or an explicit runtime
   out-dir), **never** under `src/`.

## Verification

- `python -c "import lunaris.<changed.module>"` for each touched module (no
  import error, no new cycle).
- `python -m pytest tests/ -q` (broaden when shared contracts/imports changed,
  per `AGENTS.md`).
- Grep for the symbol you moved to confirm no stale import path remains.

## Stop conditions

- The change requires a higher layer to be imported by a lower one → stop and
  propose an inversion (callback/protocol/parameter) instead.
- A "constant" already exists in `common.constants` with a different value →
  surface the conflict; do not silently add a second definition.

## Output

A short placement decision: target file/layer, the import-direction proof, any
SSOT/optional-dep adjustments, and the verification commands run.

## Acceptance

No layer-up imports; config still flows through `SimConfig`; optional deps stay
lazy in lower layers; entry points and artifact paths preserved; imports + focused
tests pass.
