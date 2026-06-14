---
name: st-lrps-evidence-audit
description: >-
  Independently audit ST-LRPS surrogate-gravity results for scientific
  defensibility before they back a claim, paper figure, or release. Use when
  asked to "audit", "sanity-check", "is this result publishable", "check for
  leakage", "is this generalization or interpolation", "verify the split",
  "force_direct vs potential_autograd", or before promoting any ST-LRPS number.
  Enforces train-only scalers, split-policy semantics, model-kind identity,
  artifact contracts, OOD/altitude evaluation, curl/energy checks, and paper-safe
  mode. NOT for training-loop engineering (use st-lrps-training), classic-SH
  physics (use astrodynamics-validation), or benchmark reproducibility plumbing
  (use reproducible-benchmarks).
---

# ST-LRPS Scientific Evidence Audit

This is an **audit**, not a defense of the run. The job is to find the reasons a
result is *not* yet defensible. Treat the implementation's intent skeptically and
report against hard gates.

## Invocation

Auto-trigger. Prefer an **isolated/independent context** (or the
`independent-review` companion) so the audit is not anchored to whoever produced
the run. Read-only: do not "fix" the run during the audit.

## Canonical sources (read before judging)

- `docs/ST_LRPS_VALIDATION_HYGIENE.md` — the authoritative hygiene contract.
- `docs/CONFIG_AND_ARTIFACT_CONTRACTS.md` — `TargetContract`, scaler/checkpoint
  provenance, runtime compatibility.
- The run dir: `config.json`, `scaler.json`, `split_manifest.json`, `checkpoints/`,
  run manifest, and any evaluation/ablation outputs.
- Full checklist: `checklists/evidence-audit.md` (read it for the gate list).

## Hard rejection conditions (any one ⇒ BLOCKER)

1. **Scaler leakage.** `scaler.json` must record `fit_scope="train_only"` with
   matching `split_seed`/index hashes. Whole-file or val-inclusive scaler fit is a
   blocker (guarded by `tests/test_st_lrps_scaler_leakage.py`).
2. **Mislabelled generalization.** `seeded_random`/`altitude_stratified` measure
   **interpolation**. Only `spatial_block` is spatial **generalization**;
   `ood_low_altitude`/`ood_high_altitude` are altitude **extrapolation**. A claim
   of "generalization" backed by an interpolation split is a blocker.
3. **Model-kind confusion.** `potential_autograd` derives `Δa = a_sign·∇ΔU`
   (no force head). `force_direct` predicts `Δa` directly and is a **separate**
   artifact — not the potential model's force head, and **not** part of the
   A0–A6 scalar ablation matrix. Conflating them is a blocker.
4. **Synthetic / quick output as evidence.** Quick-mode, synthetic, or
   reduced-scenario outputs presented as paper evidence is a blocker.
5. **Contract incompatibility.** Run's `TargetContract` (target_mode,
   baseline_kind, base_degree, target_degree, frame=`moon_fixed_cartesian`,
   derivative convention, mu/r_ref/a_sign) inconsistent with the downstream
   request is a blocker.

## Procedure

1. **Provenance.** Confirm checkpoint, scaler, split manifest, and contract all
   come from the same run and seed; confirm validation was monitor-only.
2. **Split semantics.** Read `split_manifest.json`; map the policy to the claim
   (interpolation vs spatial generalization vs altitude OOD). Check per-split
   altitude/lat/lon coverage; flag domain gaps.
3. **Metric completeness.** Field metrics + tail metrics; for force-level claims,
   explicit acceleration error, **curl** diagnostics, and **energy-drift** /
   orbit-level validation. A `force_direct` claim without curl + orbit validation
   is unsupported.
4. **OOD coverage.** Altitude extrapolation actually evaluated when the claim
   implies it; ablations (A0–A6) present for scalar-potential claims.
5. **Paper-safe consistency.** If paper-safe mode is claimed, verify scenario
   counts and report/artifact consistency (cross-check `reproducible-benchmarks`).

## Output format (required)

Group findings as: **BLOCKERS**, **MAJOR**, **MINOR**, **MISSING EVIDENCE**,
**EXPLORATORY-ONLY**, **ACCEPTABLE**. End with a one-line **VERDICT**
(`defensible` / `not yet defensible`) and the single most important next action.

## Stop conditions

- A required artifact (`scaler.json`, `split_manifest.json`, contract) is missing
  → MISSING EVIDENCE blocker; do not infer the missing provenance.

## Acceptance

Every claim is mapped to a split policy and metric that actually supports it; all
five hard gates pass; force_direct claims carry curl + orbit evidence; verdict and
next action are explicit.
