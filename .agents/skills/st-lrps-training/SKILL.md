---
name: st-lrps-training
description: >-
  Develop and run ST-LRPS surrogate-gravity training and evaluation with
  Lunaris' existing custom pipeline and artifact contracts. Use when working on
  dataset ingestion, train-only scalers, the network/loss, checkpoint selection,
  resume, ablations (A0–A6), OOD evaluation, or the potential_autograd vs
  force_direct runtimes, and on requests like "train ST-LRPS", "add a loss term",
  "why is training unstable", "set up an ablation", or "should we move training to
  Lightning". Works WITH the existing pipeline; a framework migration is a
  separate architectural evaluation, never an automatic refactor. NOT for
  auditing finished results (use st-lrps-evidence-audit) or GPU throughput of the
  propagator (use mc-backend-and-performance).
---

# ST-LRPS Training Engineering

Lunaris has a custom, contract-driven training/artifact system. The main risk
here is replacing it with a framework or breaking a provenance contract. Extend
it; do not rewrite it.

## Invocation

Auto-trigger; inline. Pair with `st-lrps-evidence-audit` once a run exists, and
`reproducible-benchmarks` when a trained model feeds a benchmark.

## Canonical sources

- `docs/DATASET_PIPELINE.md`, `docs/CONFIG_AND_ARTIFACT_CONTRACTS.md`,
  `docs/ST_LRPS_VALIDATION_HYGIENE.md`.
- Code under `src/lunaris/surrogate/st_lrps/` (`training/`, `evaluation/`,
  `paper_evidence/`).
- Entry points: `lunaris-train` (potential), `lunaris-train-force-direct`,
  `lunaris-eval`, `lunaris-eval-force-direct`, `lunaris-validate`,
  `lunaris-ablation`, `lunaris-data`.

## Invariants to preserve

1. **Train-only scalers.** Fit scalers on train indices only; never on val/test.
   `scaler.json` must keep `fit_scope="train_only"` + split provenance.
2. **Model-kind separation.** `potential_autograd` learns `ΔU` and derives
   `Δa = a_sign·∇ΔU` (no force head). `force_direct` predicts `Δa` directly and is
   a **distinct** runtime/artifact with its own student sweep — not the potential
   model's head, and not in the A0–A6 matrix.
3. **Contracts.** Respect `TargetContract` (target_mode, baseline_kind,
   base/target degree, frame `moon_fixed_cartesian`, derivative convention,
   mu/r_ref/a_sign). Don't silently change a contract field.
4. **Splits & provenance.** Use `split_manifest.json`; keep seeds, index hashes,
   and `dataset_content_sha256` intact for reproducibility.
5. **Checkpoint provenance.** Best/last selection is monitor-driven and recorded;
   resume must restore the same contract + scaler.

## Procedure

1. **Validate inputs first.** Confirm dataset contract + split manifest before
   touching the loop (`lunaris-data`, the dataset validators).
2. **Make the smallest change** to net/loss/schedule; keep device handling and
   numerical-stability guards. Mixed precision only if validated against an f64
   reference — never as a blind speedup.
3. **Diagnostics.** Watch gradient norms / NaNs; log to the existing logger; keep
   monitor-only validation (no val leakage into selection beyond the recorded
   monitor metric).
4. **Ablations / OOD** through `lunaris-ablation` and the OOD split policies, not
   ad-hoc scripts.
5. **Lightning question.** If asked to migrate to PyTorch Lightning, treat it as a
   *separate architectural evaluation* (artifact contracts, provenance,
   determinism, resume parity) — present trade-offs; do not auto-refactor.

## Verification

- `python -m pytest tests/test_mc_gpu_policy.py -q` (surrogate inference contracts)
  and the ST-LRPS training/scaler tests (`tests/test_st_lrps_scaler_leakage.py`,
  dataset/training tests under `tests/`).
- A short overfit-one-batch sanity run before any long training.

## Stop conditions

- A change would fit scalers on non-train rows, blur the two model kinds, or alter
  a contract field silently → stop.

## Output

A training change memo: what changed in net/loss/schedule, which invariants were
checked, gradient/stability diagnostics, and the validation/ablation plan.

## Acceptance

Train-only scalers and contracts preserved; model kinds kept distinct; provenance
intact; stability diagnosed; Lightning treated as evaluation not refactor; tests
pass.
