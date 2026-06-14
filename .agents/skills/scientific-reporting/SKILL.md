---
name: scientific-reporting
description: >-
  Write Lunaris technical/academic prose where every claim is traceable to a real
  artifact — result vs interpretation vs speculation kept separate. Use when asked
  to "write up the results", "draft the methods/abstract", "write a figure/table
  caption", "summarize the benchmark for the paper", "document this experiment",
  or "turn these numbers into text". States units, frame, split type, model kind,
  benchmark config, scenario count, and hardware (for runtime); never inflates
  interpolation into generalization or synthetic output into evidence. NOT for
  judging whether a result is defensible (use st-lrps-evidence-audit) or producing
  the figures themselves (use scientific-figures).
---

# Scientific Reporting (Lunaris)

The writer communicates evidence that has already passed audit. It does not
manufacture confidence: no unsupported superlatives, no invented citations or
numbers. Claims trace to artifacts.

## Invocation

Auto-trigger; inline. **Composition rule:** acceptance of evidence belongs to
`st-lrps-evidence-audit` / `reproducible-benchmarks`; this skill only *writes up*
what those have accepted. If a number isn't backed by an artifact, it doesn't go
in the prose — flag it for audit instead.

## Canonical sources

- The artifacts behind each number: run manifest, `scaler.json`,
  `split_manifest.json`, benchmark outputs, figure source data.
- `docs/BENCHMARK_RESULTS.md`, `docs/ST_LRPS_VALIDATION_HYGIENE.md`,
  `docs/REPRODUCIBLE_BENCHMARKS.md`.

## Rules

1. **Trace every number** to a file/artifact; if you can't, don't write it.
2. **Separate** result (measured) / interpretation (inference) / speculation
   (future) explicitly.
3. **Always state:** units, reference frame, split type (interpolation vs spatial
   generalization vs altitude OOD), model kind (`potential_autograd` vs
   `force_direct`), benchmark config, scenario count; **hardware** whenever runtime
   is discussed.
4. **Forbidden moves:** calling interpolation "generalization"; calling synthetic/
   quick output "evidence"; unsupported superlatives ("state-of-the-art",
   "perfect"); fabricated citations or values.
5. **Limitations** are included, not hidden; reproducibility anchors (commit,
   config, seed) are preserved.
6. **Captions are self-contained** (defer to `scientific-figures` for the figure;
   write the caption to stand alone).
7. **Terminology consistency** — one term per concept across the document.

## Procedure

1. Collect the accepted artifacts and the audit verdict for each claim.
2. Draft, tagging each sentence implicitly as result/interpretation/speculation.
3. Insert units/frame/split/model-kind/config/scenario-count/hardware at first use.
4. Add limitations + reproducibility anchors; check terminology consistency.

## Verification

- Every quantitative claim has a traceable source named in the draft or its notes.
- Grep your own draft for "generalization", "SOTA/state-of-the-art", "perfect",
  "best" — justify or remove each.
- Hand the draft's claims back through `st-lrps-evidence-audit` if any are new.

## Stop conditions

- A required number has no backing artifact, or contradicts the audit verdict →
  stop and route to audit; do not write it as established.

## Output

Prose (methods/results/caption/summary) with: traceable numbers, explicit units/
frame/split/model-kind/config/scenario-count/hardware, separated result/
interpretation/speculation, stated limitations, and reproducibility anchors.

## Acceptance

No untraceable or audit-contradicting claims; required metadata present at first
use; interpolation never called generalization; synthetic never called evidence;
limitations and anchors included.
