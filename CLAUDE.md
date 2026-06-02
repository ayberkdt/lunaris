# CLAUDE.md

This file is a lightweight repository guide for coding agents and contributors.
Keep detailed architecture notes in `docs/ARCHITECTURE.md`; this file should
stay short and operational.

## Commands

```bash
# Run CLI simulation (console entry point installed via `pip install -e .`)
lunaris --start-date 2025-01-01T00:00:00 --days 1 --alt-km 100

# Common CLI flags
--hp-km 80 --ha-km 200          # Periselene/aposelene altitudes
--inc-deg 90                    # Inclination (polar orbit)
--enable-sh --enable-srp        # Enable physics perturbations
--out-dir outputs/missions/run1 # Output directory

# Run GUI
lunaris-ui

# Run tests
pytest tests/
pytest tests/test_dynamics.py -v
```

## Architecture

The codebase is a lunar orbit propagation framework using a `src/lunaris/`
package layout, organized into four strict layers (a layer never imports from a
layer above it):

1. **`lunaris.common`** — dependency-light shared layer. `constants.py` is the
   SSOT for physical constants; `type_defs.py` holds the configuration dataclasses.
2. **`lunaris.physics`** — Numba-JIT force-model kernels (spherical harmonics,
   third-body, SRP, surface, relativity) plus `ephemeris.py` (SPICE) and the
   surrogate-gravity adapters. Never imports from `core/`.
3. **`lunaris.core`** — numerical engine: `config.py` (`SimConfig` SSOT),
   `dynamics.py` (builds the Numba RHS closure), `propagator.py`
   (`solve_ivp` → `PropagationResult`), `events.py`, and the Monte Carlo engine.
4. **`lunaris.analysis` / `lunaris.visualization` / `lunaris.ui`** —
   post-processing, reporting, Monte Carlo analysis, standalone visualization,
   and the PySide6 desktop UI (`lunaris.ui.app` + `lunaris.ui.widgets`).

The ST-LRPS surrogate-gravity pipeline lives under `lunaris.surrogate.st_lrps`.

**Configuration is a single source of truth:** everything flows through the
frozen `SimConfig` from `lunaris.core.config` (`load_default_config()` →
`apply_args_to_config()` in `lunaris.cli.main` → `cfg.validate()`). Never pass
ad-hoc kwargs.

> **Full reference:** see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the
> data flow, perturbation-flag table, external-data layout, Monte Carlo
> infrastructure (CPU/GPU backends, GPU constraints, output formats), ST-LRPS
> design, and performance notes. Keep that document as the canonical source —
> update it rather than re-describing the architecture here.

## Working Style

- Start non-trivial work by identifying the smallest safe scope and the tests
  that prove it.
- Prefer existing validators, contracts, and CLI entry points before adding new
  workflow scripts.
- Keep generated artifacts under `outputs/` or another explicit runtime output
  directory, never under `src/`.
- Preserve public CLI entry points and artifact contracts unless a task
  explicitly asks to change them.
- Run focused tests before declaring a change done; broaden tests when shared
  contracts or import behavior change.

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
