# Paper-Safe Fail Policy

Status: policy adopted 2026-07-05 (roadmap item R29a). Enforcement is being
rolled out in Sprint 2 (R29b); the table below records which rules are already
enforced in code and which are pending. This document is the single list — a
new silent-fallback path may not be added anywhere without updating it.

## The two modes

| Mode | Meaning | Failure behavior |
|---|---|---|
| `paper_safe=true` | The run's outputs may back a scientific claim, paper figure, or release artifact. | **Hard fail (raise), before scientific-looking output is produced.** No silent simplification, substitution, or downgrade of any kind. |
| `research_mode=true` (default posture for exploratory work) | The run is exploratory; convenience fallbacks are acceptable. | **Warn + fall back + record.** Every fallback is written to provenance/metadata; nothing is silent, but nothing hard-stops the run. |

The modes are mutually exclusive. A run that sets neither is treated as
research mode for fallback behavior but must still record every fallback.

## Hard-fail conditions under `paper_safe=true`

Each of these raises before producing scientific-looking outputs. In research
mode the same condition produces a warning, the stated fallback, and a
metadata record.

| # | Condition | paper_safe behavior | research_mode behavior (fallback + metadata) | Enforcement status |
|---|---|---|---|---|
| 1 | Terrain/topography provider unavailable while `impact_surface_mode` needs it | RuntimeError | Sphere fallback; `terrain_fallback=sphere` recorded | Enforced (R12; `BatchPropagationEngine._build_propagator` hard-fails when `_fallback_forbidden()`, else warns + records `terrain_fallback=sphere`) |
| 2 | Ephemeris missing/zeroed for an enabled force model | RuntimeError | Not allowed as silent zero; must warn and disable the force with a metadata record | Partially enforced (zero-ephemeris fail-closed guard exists on the propagation path) |
| 3 | Backend fallback (requested backend unavailable / unsupported physics) | RuntimeError (`allow_fallback=false` semantics) | Fall back per `backend_policy`; `fallback_applied` + `fallback_reason` recorded | Partially enforced (fallbacks always recorded + hard-fail via `sh_fallback_policy='error'` / `_fallback_forbidden()`; broader paper_safe switch pending R29b) |
| 4 | Dtype mismatch (requested vs effective/model/backend dtype) | RuntimeError | Downgrade allowed; `requested_dtype` vs `effective_dtype` recorded | Enforced at policy level (R10: `resolve_effective_dtype` single source; plan + output record `requested_dtype`/`effective_dtype`/`dtype_downgraded`); paper_safe hard-fail on downgrade pending R29b |
| 5 | ST-LRPS model-kind mismatch (artifact kind ≠ runtime expectation) | RuntimeError | Not allowed even in research mode — model identity is never substituted | Enforced (artifact contract check; benchmark `apply_paper_safe` sets `allow_contract_mismatch=false`) |
| 6 | Gravity file mismatch (artifact's `gravity_model_hash` ≠ configured gravity model) | RuntimeError | Warn + record both hashes | Partially enforced (contract compatibility report; hash coverage widening under R26) |
| 7 | Domain guard violation (state outside the surrogate's trained altitude/radius envelope) | RuntimeError (`strict_domain=true`) | Warn; sample flagged `domain_exit=true`, result marked low-confidence | Enforced in benchmark (`allow_domain_extrapolation=false`); frozen-search enforcement pending R27 |
| 8 | Synthetic / quick / legacy output modes | BenchmarkConfigError | Allowed; outputs carry the synthetic banner | Enforced (`apply_paper_safe` in `benchmark_config.py`) |
| 9 | Missing mandatory artifact metadata | Inference refuses to start | Legacy override allowed with explicit flag + metadata record | Pending (R26 artifact-contract hardening) |

## Rules of thumb

- **Never silent.** Even in research mode, a fallback that is not visible in
  provenance/metadata is a bug.
- **Fail before output.** Paper-safe failures must occur before any file that
  could be mistaken for a scientific result is written (the
  `apply_paper_safe` pattern: validate config, then run).
- **Identity is not negotiable.** Model kind, gravity model, and scaler
  provenance are never substituted in either mode; at most, research mode may
  proceed with a recorded warning where noted above.
- **Broad `except Exception` is suspect.** Any handler that cannot be
  narrowed must carry a justifying comment (R29b sweep).

## Related documents

- [backend_matrix.md](backend_matrix.md) — per-backend supported physics and
  fallback provenance fields.
- [CONFIG_AND_ARTIFACT_CONTRACTS.md](CONFIG_AND_ARTIFACT_CONTRACTS.md) —
  artifact/dataset contract rules that back conditions 5, 6, and 9.
- [REPRODUCIBLE_BENCHMARKS.md](REPRODUCIBLE_BENCHMARKS.md) — how paper-safe
  benchmarks are run and validated.
- [ST_LRPS_VALIDATION_HYGIENE.md](ST_LRPS_VALIDATION_HYGIENE.md) — split
  policies and validation posture behind paper-safe claims.
