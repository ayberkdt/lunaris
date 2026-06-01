# Perturbation Budget Analysis

`lunaris-perturbation-budget` is a mission-analysis and validation tool. It
compares instantaneous acceleration contributions, spherical-harmonic gravity
degree increments, and first-order force-model uncertainty assumptions.

It is not an electrical power analysis and it is not a new force model.

## What It Answers

- Which acceleration terms matter at selected lunar-orbit altitudes?
- How large are incremental spherical-harmonic bands such as `SH60 -> SH100`?
- When is the next gravity-degree increment smaller than SRP, albedo, thermal
  IR, tide, or combined non-gravitational uncertainty?
- What gravity degree is recommended for this configuration and threshold?

## Basic Usage

```bash
lunaris-perturbation-budget \
  --altitudes-km 50,100,300,1000,3000 \
  --inclinations-deg 0,30,60,90 \
  --true-anomalies-deg 0,90,180,270 \
  --sh-degrees 20,30,60,100,200 \
  --gravity-model path/to/lunar_gravity_model.tab \
  --out-dir outputs/perturbation_budget/default
```

For smoke tests, the command can run without a gravity model:

```bash
lunaris-perturbation-budget \
  --altitudes-km 100 \
  --inclinations-deg 0 \
  --true-anomalies-deg 0 \
  --sh-degrees 2,4 \
  --out-dir outputs/perturbation_budget/smoke
```

When no gravity model is provided, Lunaris uses deterministic synthetic
coefficients and labels the report accordingly. Synthetic coefficients are for
workflow validation only and must not be interpreted as lunar truth.

## Outputs

The output directory contains:

- `perturbation_budget.csv`: per-state acceleration contribution table.
- `gravity_degree_sensitivity.csv`: vector SH increment table.
- `force_model_uncertainty_budget.csv`: first-order model uncertainty table,
  including `Delta SH...` comparison rows against combined non-gravitational
  RSS uncertainty.
- `recommended_gravity_degree_by_altitude.csv`: derived recommendation table.
- `propagation_ablation.csv`: MVP placeholder; propagation ablation is optional.
- `runtime_budget.csv`: runtime information for the instantaneous analysis.
- `perturbation_budget_summary.md`: human-readable summary and warnings.
- `config.json`: exact configuration used.

## Interpretation

The SH degree increment is a vector difference:

```text
Delta SH60->100 = a_SH100 - a_SH60
```

The analysis decomposes acceleration vectors into the local RIC frame:

- radial: along position vector
- along-track/transverse: in the orbit plane
- cross-track/normal: along angular momentum

Force magnitude and force-model uncertainty are separate concepts. A force can
be small but poorly known, or large but modeled accurately. Recommendations use
configurable thresholds to compare SH increments with combined non-gravitational
uncertainty.

## Recommendation Caution

The recommended degree is for this configuration only. It depends on:

- altitude and orbit geometry
- mission duration and accuracy needs
- spacecraft area-to-mass ratio
- gravity model file and available degree
- Sun/Earth geometry or ephemeris
- enabled force models
- uncertainty assumptions and thresholds

Correct phrasing: "recommended degree for this configuration."

Incorrect phrasing: "SH60 is always enough above 1000 km."

## MVP Scope

The current implementation covers instantaneous acceleration budgets, SH degree
sensitivity, uncertainty budgets, recommendations, CLI output, and tests.

Propagation-level ablation and detailed runtime comparisons are intentionally
left as follow-up work because they require longer integrations and reference
model choices. A small placeholder CSV is written so report consumers can detect
that the optional step was not run.
