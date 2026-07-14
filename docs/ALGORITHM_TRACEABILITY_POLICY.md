# Algorithm Traceability Policy

This policy governs how Lunaris names, cites, classifies and traces every
scientifically or numerically meaningful algorithm, physical model,
interpolation method, integrator, sampling method, neural architecture,
optimization technique, frame/time convention, scientific data product and
project-specific heuristic.

The goal is that a developer inspecting any such implementation can answer:

1. What is the canonical name of this method?
2. Is it an exact implementation, an adaptation, an inspired method, a
   project-specific heuristic, or a delegated library implementation?
3. What is its stable Lunaris algorithm ID?
4. What is the verified primary source?
5. Which section/equation/algorithm/chapter/standard/documentation defines the
   implemented form?
6. Which Lunaris source symbols implement it?
7. Which tests validate it?
8. What assumptions and limitations apply?
9. Has Lunaris changed the original algorithm?
10. Can the source be independently verified through a DOI, ISBN, official
    report number, standard number, or stable official URL?

## The three layers

**Layer 1 - Stable Lunaris algorithm IDs.** Every entry has a permanent ID:

```
LUNARIS-<PREFIX>-<DOMAIN>-<NNN>
```

- `PREFIX`: `ALG` (named mathematical/numerical algorithm), `MODEL` (physical or
  empirical model), `HEUR` (Lunaris-specific heuristic or policy), `DATA`
  (scientific data/model product or convention), `STD` (implemented external
  standard/convention).
- `DOMAIN`: a 2-4 character subject code from the schema enum (e.g. `TB`, `SH`,
  `INT`, `EPH`, `FRM`, `ML`, `RAD`, `REL`, `TID`, `SAMP`, `DATA`, `CST`).
- `NNN`: zero-padded sequential number within (prefix, domain).

**ID stability is a hard rule.** Once an ID is merged it is never renumbered,
reused, or deleted. Retire an obsolete entry by setting `status: retired`; it
stays in the registry. Do not renumber existing entries to improve ordering.

**Layer 2 - Canonical bibliographic citation keys.** Verified BibTeX records in
[`references/references.bib`](../references/references.bib), keyed
`FirstAuthorYYYYShortMnemonic` (e.g. `Battin1999Astrodynamics`,
`Kahan1965ReducingErrors`, `Sitzmann2020SIREN`). Citation keys may change if
metadata is corrected; IDs never do.

**Layer 3 - Human-readable canonical names.** The name recognised in the primary
literature or authoritative documentation (e.g. "Kahan compensated summation",
"Spherical linear interpolation of unit quaternions (SLERP)"). Banned vague
adjectives (`advanced`, `optimized`, `professional`, `improved`, `enhanced`)
unless an exact technical definition follows in `notes`; this is enforced by
`tools/algorithm_registry.py`.

## Classification guide (`implementation_class`)

| Class | Meaning | Example in registry |
| --- | --- | --- |
| `exact` | Implements the published method as-is | `LUNARIS-ALG-SUM-001` (Kahan) |
| `exact_reformulation` | Algebraically equivalent, better conditioned | `LUNARIS-ALG-TB-001` (Battin F(q)) |
| `adaptation` | Published method adapted to a new setting | `LUNARIS-MODEL-TID-001` (Earth tide formalism applied to the Moon) |
| `inspired` | Motivated by a method but not a faithful implementation | (use only with an honest note) |
| `heuristic` | Lunaris-specific policy; may have no external source | `LUNARIS-HEUR-SH-001` (pole-stable truncation) |
| `delegated_library` | Method provided by a third-party library | `LUNARIS-ALG-INT-001` (SciPy DOP853) |
| `standard_implementation` | Faithful implementation of a textbook/standard method | `LUNARIS-ALG-SH-002` (SH acceleration) |

## Citation-verification procedure

Before an entry may claim a verified source:

1. Find the primary source (paper, book, standard, data product).
2. Confirm a persistent identifier resolves: DOI, ISBN, report number, standard
   number, or a stable official URL. **Never invent metadata**; never derive a
   year from a secondary source.
3. Confirm the section/equation/chapter that defines the *implemented* form, and
   record what was checked, where, and when in
   `primary_reference.verification_notes`.

`verification_status` values and when each is allowed:

| Status | Meaning |
| --- | --- |
| `verified_primary_source` | Identifier resolves AND the defining section/chapter/equation/pages is confirmed. Requires a locator. |
| `verified_secondary_source` | Confirmed via an authoritative secondary source only |
| `identifier_verified_content_pending` | Identifier (DOI/ISBN/URL) confirmed, but the exact defining section/equation is not yet pinned |
| `pending_verification` | Not yet verified |
| `unverifiable` | Lunaris-specific; no external source exists (typical for `HEUR`) |

Fail closed: if you cannot confirm the defining section/equation, do **not** use
`verified_primary_source`.

## Adding or changing an entry

1. Edit [`docs/algorithms/algorithm_registry.yaml`](algorithms/algorithm_registry.yaml)
   (and `references/references.bib` if a new source is needed).
2. Run `python tools/algorithm_registry.py validate`.
3. Run `python tools/algorithm_registry.py generate` to refresh
   [`docs/ALGORITHM_CATALOG.md`](ALGORITHM_CATALOG.md).
4. Commit all changed files together. CI runs
   `tests/test_algorithm_registry.py`, which fails on schema violations,
   referential-integrity errors, unresolved symbols, missing test paths, or a
   stale catalogue.

`python tools/algorithm_registry.py audit` is an authoring aid (not run in CI):
it lists `src/lunaris` files with algorithm-ish keywords that no entry covers.

## Naming bugs vs numerical bugs

This system is documentation/traceability first. If identification reveals a
misleading name (for example a token that implies a different method than the
code implements), record the correct canonical name and a note in the registry
**without changing numerical behaviour**. Any numerical change is a separate,
clearly-labelled, user-approved commit. Two such naming findings are already
recorded (the `RK8` token is a Gragg-Bulirsch-Stoer extrapolation method, and
`Y6`/`Y8` are the recursive triple-jump construction rather than Yoshida's
optimized coefficient sets).

## Relationship to other files

- [`CITATION.cff`](../CITATION.cff) - how to cite **Lunaris the software**. It is
  not the algorithm bibliography.
- [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) - third-party **license**
  inventory. Distinct from algorithm traceability.
- `references/references.bib` - the algorithm/model/data/standard bibliography
  (this system's Layer 2).

## What does NOT need an entry

Pure engineering plumbing (chunking, storage, progress reporting, Qt widgets),
trivial arithmetic helpers, and UI styling do not need registry entries. Folklore
numerical hygiene (clamping, angle wrapping) may be omitted or grouped. When in
doubt, an entry is cheap and honest; a vague name is not.

## Coverage status

The registry covers the core force models (third-body, spherical-harmonic
gravity, Earth-J2, solid tides, relativity, radiation pressure), interpolation
and frames (Catmull-Rom, SLERP, Hamilton quaternion rotation, SPICE/DE440, lunar
PA frame), integrators and event handling, sampling, orbital-element
conversions, the ST-LRPS neural architectures and optimization, scientific data
products (GRAIL GL1800F, CODATA 2018), and the phase-drift diagnostic.
Deliberately not yet itemised (candidates for future entries): ensemble UQ
covariance statistics, frozen-orbit search, generic bilinear/nearest grid
sampling, and terrain-aware impact DEM lookup. Add them following the procedure
above; never remove an existing ID.
