---
name: reproducible-benchmarks
description: >-
  Run and validate Lunaris gravity/propagation benchmarks so the numbers are
  reproducible and paper-defensible. Run ONLY when the user explicitly asks to
  "run the benchmark", "regenerate benchmark results", "produce paper evidence",
  or "validate this benchmark for the paper" — this is an expensive,
  state-producing workflow, so do not auto-trigger on casual benchmark mentions.
  Covers lunaris-benchmark / lunaris-st-lrps-paper-evidence config selection,
  provenance capture, scenario-count and unit checks, artifact-contract
  compatibility, and report consistency. NOT for auditing an already-produced
  result's science (use st-lrps-evidence-audit) or generic timing (use
  mc-backend-and-performance).
---

# Reproducible Benchmarks & Paper Evidence

A benchmark number is only worth as much as its provenance. This workflow
produces results that can be regenerated and that survive review.

## Invocation

**Manual / explicit only.** This is a side-effecting, potentially long workflow
(`lunaris-benchmark`, `lunaris-st-lrps-paper-evidence`). Never start it just
because "benchmark" appears in a question. Confirm the config and output path
before running anything heavy.

## Canonical sources

- `docs/REPRODUCIBLE_BENCHMARKS.md` — the entry point, config schema, and
  artifact-contract compatibility policy.
- `docs/BENCHMARK_RESULTS.md` — what current published numbers say.
- `configs/benchmarks/*.json` — committed benchmark configs (JSON; YAML accepted
  if PyYAML present). Local checkpoint paths are passed at runtime, never hardcoded.
- Entry points: `lunaris-benchmark` (`...evaluation.compare_gravity_models`),
  `lunaris-st-lrps-paper-evidence` (`...paper_evidence.runner`).

## Procedure

1. **Pick a committed config**, not ad-hoc flags:
   `lunaris-benchmark --config configs/benchmarks/<name>.json --model-dir <run> --out <outdir>`.
2. **Capture provenance before running:** current commit (`git rev-parse HEAD`,
   dirty/clean), Python/torch/numba versions, GPU name, and the exact command.
   Record hardware whenever a runtime number will be reported.
3. **Contract compatibility.** When `surrogate.model_dir` is set, the benchmark
   builds a requested `ArtifactContract` and compares it (strict by default) to the
   selected run. A mismatch is a hard stop — do not relax to "loose" to force it.
4. **Scenario integrity.** Verify scenario seed, count, sampling family, and
   altitude envelope match the config; check scenario IDs and that the count is
   the intended (not a quick-mode subset). Units must be SI; model names must be
   the canonical ones.
5. **Output validation.** Confirm the required output files exist; check for
   accidental truth duplication (a "surrogate" column that is actually the truth
   model); confirm report values match the underlying artifact values.
6. **Paper trail.** Keep the config, provenance, command, and outputs together
   under `outputs/` (never under `src/`).

## Verification

- Re-run with the same config + seed → identical scenario IDs and matching
  numbers (within recorded tolerances).
- `lunaris-validate` for the model when validation evidence is part of the claim.
- Cross-check the produced numbers with `st-lrps-evidence-audit` before they back
  a paper claim.

## Stop conditions

- Artifact contract mismatch, missing provenance, or a quick/synthetic config
  presented as final evidence → stop and report; do not publish the number.
- The user has not confirmed an explicit config/output path → do not launch a
  long run.

## Output

A benchmark record: config used, full provenance (commit, env, hardware, command),
scenario integrity check, contract-compatibility result, output-file inventory,
and a consistency check between report and artifacts.

## Acceptance

Result regenerates from the committed config + seed; provenance is complete;
contracts strict-match; scenario counts/units/names verified; report matches
artifacts; outputs live under `outputs/`.
