# Versioning and Support Policy

## Version scheme

Lunaris follows **semantic versioning** (`MAJOR.MINOR.PATCH`, PEP 440 forms
for pre-releases: `0.1.0rc1`).

While the version is **0.x**, the semver contract is:

- **PATCH** (0.1.0 -> 0.1.1): bug fixes only; no API, CLI, or on-disk schema
  changes; numerical results change only when the previous number was a bug.
- **MINOR** (0.1 -> 0.2): may add features and may make breaking changes, but
  every breaking change must be listed under a "Breaking" heading in
  [CHANGELOG.md](../CHANGELOG.md).
- Pre-releases (`b`, `rc`) may change before the final release, but changes to
  a documented Python tier must be recorded in the changelog and API snapshot.

From **1.0.0** on, breaking changes to stable surfaces require a MAJOR bump.

## What is a stable surface

The stability classification of every console entry point and documented
Python module lives in [PUBLIC_API.md](PUBLIC_API.md). That document is the
canonical human-readable inventory; `docs/public_api_manifest.json` is the
machine-readable Python-module inventory. In addition:

- **Stable:** documented CLI flags of stable entry points; the config-file
  schemas under `configs/`; the output-artifact contracts listed in
  [CONFIG_AND_ARTIFACT_CONTRACTS.md](CONFIG_AND_ARTIFACT_CONTRACTS.md); and the
  `user-stable` Python modules listed in PUBLIC_API.md.
- **Documented provisional:** the `documented-provisional` Python modules in
  PUBLIC_API.md. They are supported direct-import surfaces, but while Lunaris
  is 0.x they may change at a MINOR release with a changelog entry and the
  compatibility process below.
- **Cross-subsystem internal:** non-underscored implementation contracts used
  across Lunaris subsystem boundaries. They are tracked in the API manifest
  and snapshot for review, but are not a supported downstream API.
- **Versioned, migratable:** on-disk artifact schemas carry explicit schema
  versions (`st_lrps_checkpoint_v2`, `st_lrps_run_manifest_v1`,
  `st_lrps_compute_accounting_v1`, dataset contract). Readers accept the
  current schema version and the documented legacy fields (for example the
  `content_sha256` / `dataset_sha256` dataset-hash normalization); when a
  schema version is retired, the last release that reads it is named in the
  changelog.
- **Not stable:** Python modules not listed in PUBLIC_API.md, research-preview
  features (ST-LRPS accuracy characteristics), experimental backends (CUDA),
  and anything under `outputs/`.

## Deprecation process

1. The feature, flag, or Python name emits a deprecation warning when practical
   and is marked in the changelog and PUBLIC_API.md.
2. It keeps working for at least one MINOR release after the warning appears.
3. Removal is listed under "Breaking" in the changelog.

Low-level Numba aliases that cannot warn without changing compiled call
semantics are documented compatibility exceptions: they remain importable and
listed in the API snapshot for the same retention period.

## Support expectations

- Only the **latest release** receives fixes; there are no long-term support
  branches at 0.x.
- Security reports: see [SECURITY.md](../SECURITY.md). Security fixes are
  released as the next PATCH on the latest release.
- Reproducibility aids per release: hash-pinned lock files (`locks/`), the
  data-manifest SHA-256 digests, and provenance blocks (git commit, config
  hash, dataset hash) recorded in run manifests.

## Upgrade checklist for operators

1. Read the changelog section for the target version (especially "Breaking").
2. Upgrade in a staging environment from the pinned lock files.
3. Re-run `lunaris-data verify --strict --runtime`.
4. Re-run your acceptance scenarios; compare run-manifest provenance blocks
   (backend, config hash, dataset hash) against the previous baseline before
   trusting new numbers.
