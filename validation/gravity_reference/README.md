# Gravity Reference Validation

This directory holds immutable reference contracts for external validation of
Lunaris lunar spherical-harmonic gravity.

The validation is split into two questions:

- **Field level:** at fixed Moon-body coordinates, does Lunaris evaluate the same
  potential and acceleration as an independent reference?
- **Trajectory level:** with the same initial state, frame, time scale, gravity
  model, integrator, and disabled-force contract, does Lunaris reproduce an
  independent gravity-only trajectory?

Both layers are active. See `docs/GRAVITY_ENGINE_EXTERNAL_VALIDATION.md` for the
full status and the honesty note on why no NASA gravity-only *truth* trajectory
exists.

## Committed Benchmarks

**Field**

- `benchmarks/field/synthetic_degree4_oracle.json` — Lunaris vs an independent
  from-scratch direct-formula oracle on a synthetic degree-4 fixture (~1e-12).
  Low-degree correlated-error guard; no external tools.
- `benchmarks/field/grail_degree120_pyshtools_oracle.json` — Lunaris vs
  **pyshtools** on the real GRAIL JGGRX_1800F model at degree 120 (machine
  precision). Primary high-degree external truth.

The degree-120 internal direct-formula oracle benchmark was removed because its
classical Legendre recurrence is unstable above degree ~80; pyshtools is the
trustworthy high-degree reference.

**Trajectory**

- `benchmarks/trajectory/grail_degree32_pyshtools_trajectory.json` — Lunaris'
  production propagator vs an independent gravity-only arc (pyshtools force +
  SciPy `DOP853`). Non-rotating field; ~5e-8 m position agreement over ~2
  orbits. Rotating body-fixed fields fail closed.

Regenerate (needs pyshtools; not run in normal CI):

```bash
python validation/gravity_reference/generators/field/generate_grail_degree120_pyshtools.py
python validation/gravity_reference/generators/trajectory/generate_grail_degree32_pyshtools.py
```

Run the committed benchmarks (no external tools, no network):

```bash
lunaris-validate gravity-field      --manifest benchmarks/field/grail_degree120_pyshtools_oracle.json      --out outputs/validation/gravity_reference/grail_degree120_pyshtools
lunaris-validate gravity-trajectory --manifest benchmarks/trajectory/grail_degree32_pyshtools_trajectory.json --out outputs/validation/gravity_reference/grail_degree32_trajectory
```

Normal CI must not download GRAIL files, install GMAT/Orekit/Tudat, or touch
network resources; it consumes the committed reference bytes only.

## Reference Data Policy

- Preserve original external bytes when redistribution is allowed.
- Record SHA-256 for every reference, coefficient file, and generator script.
- Do not edit numerical reference values by hand.
- Generated run outputs belong under `outputs/validation/gravity_reference/`.
- Mission SPKs/OEMs are observational comparisons unless the source explicitly
  proves a gravity-only dynamics contract.

