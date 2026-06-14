---
name: independent-review
description: >-
  Run an adversarial, independent review of a Lunaris change set — find problems,
  don't defend the author. Use when asked to "review this PR/diff independently",
  "red-team this change", "what could go wrong here", "is this ready to merge", or
  for a second-opinion pass before merge. Inspects the diff for hidden
  assumptions, missing tests, scientific-claim overreach, frame/unit/sign risks,
  UI/accessibility regressions, unfair performance comparisons, doc/output-contract
  drift, and unsafe or destructive behavior. Produces BLOCKER/MAJOR/MINOR/MISSING
  EVIDENCE/POSITIVE/VERDICT. NOT for applying fixes (it reviews, it does not edit)
  or for ST-LRPS result science specifically (use st-lrps-evidence-audit).
---

# Independent Adversarial Review

The value here is independence: review as if you did **not** write the change.
Look for what's wrong or unproven, not reasons it's fine. Do not modify code
during the review.

## Invocation

Auto-trigger on explicit review requests. Prefer an **isolated context / separate
subagent** so the review is not anchored to the implementation conversation.
Compose with `astrodynamics-validation`, `st-lrps-evidence-audit`,
`accessibility-audit`, and `mc-backend-and-performance` for domain depth.

## Inputs

- The diff: `git diff` / `git diff --stat` (and `git status --short`).
- The touched files and their tests; relevant `docs/` contracts.

## What to hunt for

1. **Hidden assumptions** — unstated preconditions, silent defaults, edge cases
   (empty/NaN/degree 0/single sample).
2. **Missing tests** — new behavior without a test; changed contract without a
   broadened test (`AGENTS.md` test policy).
3. **Scientific overreach** — claims stronger than the evidence; interpolation
   framed as generalization; synthetic shown as real.
4. **Frame / unit / sign risks** — inertial↔fixed, quaternion conjugate, SI units,
   potential-vs-acceleration sign (defer detail to `astrodynamics-validation`).
5. **Backend honesty** — requested vs actual backend, silent physics drop, CPU run
   mislabeled as GPU, unfair benchmark (warm-up, dtype, mismatched config).
6. **UI / accessibility regressions** — hard-coded colors, UI-thread blocking,
   broken focus/resize, contrast.
7. **Drift** — docs, output/artifact contracts, or entry-point signatures changed
   without updating dependents.
8. **Safety** — destructive/remote actions, secrets, broad permissions, deletion
   of user data.

## Procedure

1. Read the full diff; map each change to its blast radius.
2. For each hunt category, state a concrete finding or "checked, clear".
3. Verify claims against the code, not the author's description.
4. Note positives too (so the verdict is calibrated).

## Output format (required)

```
BLOCKER:        <must fix before merge>
MAJOR:          <should fix>
MINOR:          <nice to fix>
MISSING EVIDENCE: <claims/areas not proven>
POSITIVE FINDINGS: <what is genuinely good>
VERDICT:        approve / approve-with-nits / request-changes / block
```

## Stop conditions

- The diff cannot be obtained → say so; do not review from description alone.

## Acceptance

Independent (not defensive); every hunt category addressed (finding or "clear");
findings are specific and located; output uses the required format with an explicit
verdict; no code was modified during the review.
