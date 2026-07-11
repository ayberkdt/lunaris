# Versioning and Support Policy

## Version scheme

Lunaris follows **semantic versioning** (`MAJOR.MINOR.PATCH`, PEP 440 forms
for pre-releases: `0.1.0rc1`).

While the version is **0.x**, the semver contract is:

- **PATCH** (0.1.0 → 0.1.1): bug fixes only; no API, CLI, or on-disk schema
  changes; numerical results change only when the previous number was a bug.
- **MINOR** (0.1 → 0.2): may add features and may make breaking changes, but
  every breaking change must be listed under a "Breaking" heading in
  [CHANGELOG.md](../CHANGELOG.md).
- Pre-releases (`b`, `rc`) make no stability promises beyond what the README
  states.

From **1.0.0** on, breaking changes to stable surfaces require a MAJOR bump.

## What is a stable surface

The stability classification of every console entry point lives in
[PUBLIC_API.md](PUBLIC_API.md) — that document is the canonical inventory
(enforced by `tests/test_repo_hygiene.py`). In addition:

- **Stable:** documented CLI flags of stable entry points; the config-file
  schemas under `configs/`; the output-artifact contracts listed in
  [CONFIG_AND_ARTIFACT_CONTRACTS.md](CONFIG_AND_ARTIFACT_CONTRACTS.md).
- **Versioned, migratable:** on-disk artifact schemas carry explicit schema
  versions (`st_lrps_checkpoint_v2`, `st_lrps_run_manifest_v1`,
  `st_lrps_compute_accounting_v1`, dataset contract). Readers accept the
  current schema version and the documented legacy fields (e.g. the
  `content_sha256` / `dataset_sha256` dataset-hash normalization); when a
  schema version is retired, the last release that reads it is named in the
  changelog.
- **Not stable:** Python module paths and function signatures (the Python API
  is not yet a supported surface — drive Lunaris through the CLIs and config
  files), research-preview features (ST-LRPS accuracy characteristics),
  experimental backends (CUDA), and anything under `outputs/`.

## Deprecation process

1. The feature/flag emits a deprecation warning and is marked in the changelog
   and PUBLIC_API.md.
2. It keeps working for at least one MINOR release after the warning appears.
3. Removal is listed under "Breaking" in the changelog.

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
