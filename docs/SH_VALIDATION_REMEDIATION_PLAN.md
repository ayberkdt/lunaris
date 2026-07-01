# Spherical-Harmonic Convention Validation Record

This file is an audit trail for the lunar spherical-harmonic convention fix. It
is not the primary ST-LRPS performance narrative and should not be cited as a
benchmark result.

## Current Invariant

Lunaris uses the GRAIL/geodesy spherical-harmonic convention:

- 4pi fully normalized coefficients.
- No Condon-Shortley phase.
- Dataset metadata records
  `spherical_harmonic_convention="4pi_geodesy_no_condon_shortley_v1"`.
- Dataset metadata records
  `gravity_label_engine_version="lunaris_sh_v2"`.

The runtime engine, ST-LRPS label generator, torch/GPU evaluator, and
independent validation harnesses are expected to agree on that convention.

## Closed Validation Items

The following items are complete and covered by tests:

- ST-LRPS label generation matches `GravityModel.accel_fixed` on odd-order
  tesseral/sectoral fields.
- Dataset and artifact contracts carry the SH convention and label-engine
  version, so missing/stale metadata fails closed unless an explicit legacy
  override is requested.
- Torch/GPU spherical-harmonic evaluation matches the validated numba engine on
  odd-order fields at degrees used by benchmark paths.
- Frame and field invariants cover zonal z-rotation equivariance, zero
  longitudinal acceleration for zonal-only fields, and monopole radial behavior.
- External-reference validation checks the in-repo independent oracle and the
  optional `pyshtools` path without reusing the production recurrence.

Relevant tests include:

- `tests/test_st_lrps_generator_phase_parity.py`
- `tests/test_dataset_contract.py`
- `tests/test_artifact_contract.py`
- `tests/test_sh_frame_invariants.py`
- `tests/test_independent_sh_validation.py`

## Documentation Rule

Do not lead user-facing docs with the old convention mistake. The current state
is simply:

> Lunaris uses the no-Condon-Shortley GRAIL/geodesy convention, records that
> convention in generated datasets/artifacts, and rejects incompatible artifacts
> in paper-safe benchmark paths.

Historical details belong in this audit file only.

## Benchmark Implication

Pre-alignment ST-LRPS datasets, checkpoints, truth trajectories, and active
accuracy tables are not accepted benchmark evidence. Current docs should publish
new orbit-accuracy numbers only after dataset regeneration, retraining, and a
paper-safe benchmark run with passing `validation_report.json` and complete
`benchmark_manifest.json`.

The active reporting contract is
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

## Remaining External Checks

These are optional external validation extensions, not blockers for the current
in-repo convention invariant:

1. Install `tudatpy` and run the gated C++ toolkit cross-check.
2. Generate and freeze a physical rotating-frame gravity-only trajectory
   reference using the manifest-backed NAIF lunar kernel set
   (`naif0012.tls`, `moon_de440_250416.tf`, `moon_pa_de440_200625.bpc`, and the
   DE440 support kernels).
3. Extend the optional `pyshtools` statistical field validation to degree 200 if
   a future paper claim needs that external degree coverage.
