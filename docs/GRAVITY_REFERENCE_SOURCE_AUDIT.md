# Gravity Reference Source Audit

Audit date: 2026-06-17.

This audit looked for public reference material suitable for validating
Lunaris's lunar spherical-harmonic gravity field and gravity-only propagation.
The result is field-first: no complete public lunar gravity-only trajectory was
accepted during this audit.

## Summary Answers

- Public, complete lunar gravity-only truth trajectory found: **No**.
- Public pointwise lunar gravity validation vectors found: **No accepted vector
  set in this audit**.
- Independent high-precision implementation found: **Candidate tools exist**
  (GMAT/Orekit/Tudat), but no ready immutable Lunaris-compatible vector/arc was
  accepted.
- Normal use requires GMAT/Orekit/Tudat installation: **No**.
- First implementation route: **Layer A synthetic field benchmark with an
  independent direct-formula oracle**, plus fail-closed trajectory scaffolding.

## Candidate Sources

| Organization | Project | URL | Reference class | Moon-specific | Decision |
|---|---|---|---|---|---|
| NASA Technology Transfer | GMAT | https://software.nasa.gov/software/GSC-18094-1 | external_tool_generated_trajectory candidate | Tool supports lunar missions, but page is not a reference artifact | Candidate generator only |
| GMAT development team / SourceForge | GMAT R2026a code and releases | https://sourceforge.net/projects/gmat/ and https://sourceforge.net/p/gmat/git/ci/GMAT-R2026a/tree/ | external_tool_generated_trajectory candidate | Tool/data may support Moon gravity | No committed public lunar gravity-only arc accepted |
| NASA PDS Geosciences / JPL | GRAIL LGRS SHADR products | https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/ | incomplete_reference for vectors; coefficient source | Yes | Suitable coefficient source, not pointwise/traj truth |
| NASA PDS / JPL | JGGRX_1800F label | https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/jggrx_1800f_sha.lbl | incomplete_reference for vectors | Yes | Provides model contract: degree/order, normalization, radius, DE440 PA frame notes |
| NAIF/JPL | Generic SPK planet kernels | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/ | mission_ephemeris_non_gravity_only | Moon ephemerides/orientation context | Not gravity-only truth |
| NAIF/JPL | Generic PCK/BPC kernels | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/ | mission_ephemeris_non_gravity_only | Moon PA BPCs available | Orientation data, not field validation vectors |
| Orekit | Forces documentation | https://www.orekit.org/site-orekit-latest/architecture/forces.html | independent_high_precision_field_oracle candidate | General central gravity | Candidate implementation source, no accepted lunar vectors found |
| TudatPy | Acceleration API | https://py.api.tudat.space/en/latest/dynamics/propagation_setup/acceleration.html | external_tool_generated_trajectory candidate | General spherical-harmonic support | Candidate generator only |

## GRAIL SHADR Notes

The PDS GRAIL SHADR directory lists multiple lunar gravity coefficient products,
including `jggrx_1800f_sha.tab`. The corresponding label states that GL1800F is
a JPL lunar gravity field to degree and order 1800, uses fully normalized
coefficients, has reference radius 1738.0 km, and uses DE440 to define the lunar
body-fixed principal-axes coordinate system. That is sufficient to identify a
coefficient contract, but it is not a pointwise validation vector set and not a
gravity-only trajectory.

## Rejected Or Report-Only Classes

Mission ephemerides and reconstructed spacecraft trajectories are not accepted
as gravity-engine truth unless the source explicitly provides the gravity-only
dynamics contract. SPK/OEM products usually include orbit determination,
maneuvers, third bodies, radiation pressure, and other modeled or estimated
effects.

## Selected Route

The committed implementation uses:

- a small synthetic degree-4 coefficient fixture;
- a deterministic body-fixed point set covering equatorial, mid-latitude,
  near-pole, sectoral/tesseral, and high-altitude cases;
- an independent direct-formula potential evaluator with finite-difference
  acceleration;
- manifest-declared hashes and thresholds;
- report/provenance outputs under `outputs/validation/gravity_reference/`.

This is not official NASA/JPL validation evidence. It is a reproducible
framework smoke benchmark that prevents correlated implementation errors and
prepares the path for later accepted GRAIL/GMAT/Orekit/Tudat artifacts.

