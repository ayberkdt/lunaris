# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
