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

## Ephemeris Interpolation Error Budget

The runtime ephemeris (`lunaris.physics.ephemeris`) samples SPICE at a uniform
`output_dt_s` grid (default 60 s) and interpolates between nodes:

- **Sun/Earth state**: schema-v2 tables store SPICE `spkezr` position and
  velocity samples in SI units and use cubic Hermite interpolation. Position
  and velocity are exact at every node and the interpolant is C¹ (not generally
  C²). The interpolation error of a cubic on step `h` is bounded by
  `~h⁴·max|d⁴r/dt⁴|/384`, with `|d⁴r/dt⁴| ≈ r·ω⁴` for near-circular apparent
  motion at angular rate ω. With `h = 60 s`:
  - Earth position in the Moon-centered frame (r ≈ 3.84e8 m,
    ω ≈ 2.7e-6 rad/s): ≲ 1e-9 m.
  - Sun position (r ≈ 1.50e11 m, ω ≈ 2.5e-6 rad/s synodic): ≲ 2e-4 m —
    relative error ~1e-15, at the float64 representation limit.

  Direct DE440/SPICE validation at the default 60 s cadence (1000 seeded
  off-node queries over one day) measured maximum Hermite position errors of
  6.51e-5 m for Earth and 1.87e-3 m for the Sun. The retained position-only
  Catmull-Rom compatibility path measured 4.85e3 m and 1.39e5 m maxima because
  its clamped endpoint tangent is not the SPICE state derivative. In the same
  matched one-day low-lunar-orbit DOP853 comparison, Hermite required 4190 RHS
  evaluations versus 4250 for Catmull-Rom, and the final positions differed by
  2.92e-3 m. These are configuration-specific validation results, not general
  error bounds; the reproducible payload and kernel hashes are in
  `validation/ephemeris/interpolation_validation_2026_07_18.json`.
- **Moon orientation**: per-interval quaternion SLERP, which is exact for a
  constant rotation rate about a fixed axis over each 60 s interval and
  therefore assumes libration is linear within an interval. Physical libration
  (amplitude ~1e-4–5e-4 rad, period ≥ 27.3 d) gives an angular acceleration
  ≈ A·ω² ≲ 4e-15 rad/s², i.e. an intra-interval orientation error
  ≲ 1e-12 rad and a surface-projected position error ≲ 1e-5 m. The default
  `output_dt_s = 60 s` is ~4e4× shorter than the libration period, so the
  constant-rate assumption holds with large margin.
- **Caveat for high-order adaptive integrators**: cubic Hermite is C¹ but not C²
  at the table nodes. Integrators whose error estimators sample higher
  derivatives (e.g. DOP853) can react to the C² discontinuity with extra step
  rejections near node crossings. This affects step-size efficiency, not
  accuracy, and only matters when the adaptive step grows beyond the node
  spacing. This is a measured performance concern, not a claim of C²
  smoothness. Legacy serialized position-only archives are rejected rather
  than silently resuming with a different interpolant.

These bounds are analytic order-of-magnitude estimates for the default
cadence; halving `output_dt_s` scales the position-interpolation terms by
1/16 (h⁴).

## MVP Scope

The current implementation covers instantaneous acceleration budgets, SH degree
sensitivity, uncertainty budgets, recommendations, CLI output, and tests.

Propagation-level ablation and detailed runtime comparisons are intentionally
left as follow-up work because they require longer integrations and reference
model choices. A small placeholder CSV is written so report consumers can detect
that the optional step was not run.
