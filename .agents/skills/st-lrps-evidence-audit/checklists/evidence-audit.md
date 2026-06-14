# ST-LRPS Evidence Audit Checklist

Authoritative source: `docs/ST_LRPS_VALIDATION_HYGIENE.md` and
`docs/CONFIG_AND_ARTIFACT_CONTRACTS.md`. This checklist operationalizes them.
Audit independently — look for why the result is *not* defensible.

## A. Hard gates (any failure ⇒ BLOCKER)
- [ ] **Train-only scalers.** `scaler.json` has `fit_scope="train_only"`,
      `split_policy`, `split_seed`, and matching `train/val/test_index_hash`. No
      whole-file / val-inclusive fit.
- [ ] **Split ↔ claim match.** `seeded_random` / `altitude_stratified` =
      interpolation; `spatial_block` = spatial generalization;
      `ood_low_altitude` / `ood_high_altitude` = altitude extrapolation. A claim
      must be backed by the matching policy in `split_manifest.json`.
- [ ] **Model-kind identity.** `potential_autograd` → `Δa = a_sign·∇ΔU`, no force
      head. `force_direct` → predicts `Δa` directly, separate artifact, own student
      sweep, NOT in the A0–A6 matrix. No conflation.
- [ ] **No synthetic/quick-as-evidence.** Quick-mode / synthetic / reduced-scenario
      output is never presented as paper evidence.
- [ ] **Contract compatibility.** `TargetContract` fields (`target_mode`,
      `baseline_kind`, `base_degree`, `target_degree`, frame
      `moon_fixed_cartesian`, derivative convention, `mu_si`, `r_ref_m`, `a_sign`)
      are consistent with the downstream request.

## B. Provenance
- [ ] Checkpoint, scaler, split manifest, contract all from the same run + seed.
- [ ] Validation was monitor-only (no leakage into selection beyond the recorded
      monitor metric).
- [ ] `dataset_content_sha256` / `dataset_contract_hash` present and consistent.

## C. Metrics
- [ ] Field metrics + tail metrics reported.
- [ ] Force-level claims include explicit **acceleration error**, **curl**
      diagnostics, and **energy-drift** / orbit-level validation.
- [ ] `force_direct` claims include curl + orbit validation (no autograd at
      inference, so curl is not guaranteed).

## D. Domain coverage
- [ ] Per-split altitude / latitude / longitude ranges cover the claimed regime.
- [ ] Altitude OOD evaluated when the claim implies extrapolation.
- [ ] A0–A6 ablations present for scalar-potential claims.

## E. Paper-safe consistency
- [ ] If paper-safe mode is claimed: scenario counts, model naming, and
      report↔artifact values are consistent (hand to `reproducible-benchmarks`).

## Output grouping
BLOCKERS / MAJOR / MINOR / MISSING EVIDENCE / EXPLORATORY-ONLY / ACCEPTABLE,
then a one-line VERDICT (`defensible` / `not yet defensible`) and the single most
important next action.
