# Deployment Guide (Enterprise / Offline)

How to install, provision, and operate Lunaris in a managed environment —
including machines with no internet access. For a 10-minute developer setup see
[GETTING_STARTED_10_MINUTES.md](GETTING_STARTED_10_MINUTES.md); this document
covers the operational concerns around it.

## 1. What you are deploying

- A Python package (`lunaris`, wheel or editable checkout), CPython 3.10–3.12.
- Optional extras: `ml`/`hpc` (PyTorch, h5py, psutil), `ui` (PySide6 desktop
  apps), `reports` (PDF export). Headless servers should install `[hpc]` only —
  this also avoids the LGPL surface of Qt (see
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).
- External mission data (SPICE kernels, gravity coefficients, optionally
  topography/albedo/thermal) — **not bundled**, provisioned via `lunaris-data`.

Supported platforms and the release posture (ST-LRPS = research preview,
CUDA = experimental) are stated in the [README](../README.md); those labels are
part of the deployment contract — do not present surrogate output as validated
results.

## 2. Online install

```bash
python -m venv lunaris-env && source lunaris-env/bin/activate
pip install lunaris[hpc]          # headless; use [all] for UI workstations
lunaris-data download --preset full-gravity
lunaris-data verify --strict --runtime
```

`lunaris-data` downloads only from the official providers recorded in the
manifest (NAIF/JPL, NASA PDS), refuses non-HTTPS URLs, writes atomically, and
checks the recorded SHA-256 digest of every entry that has one.

## 3. Offline / air-gapped install

1. **Wheels.** On a connected machine, download the wheel set:
   `pip download lunaris[hpc] -d wheelhouse/` (add
   `--index-url https://download.pytorch.org/whl/cpu` for the torch wheel).
   Transfer `wheelhouse/` and install with
   `pip install --no-index --find-links wheelhouse lunaris[hpc]`.
2. **Data mirror.** Run `lunaris-data download` on the connected machine, then
   transfer the resulting data root. Alternatively host an internal mirror and
   point a copy of the manifest at it: `lunaris-data download
   --manifest internal_data_sources.json`. Keep the `sha256` fields — they are
   verified regardless of the URL host.
3. **Data root.** Set `LUNARIS_DATA_DIR=/srv/lunaris-data` (or pass
   `--data-dir`) — recommended for managed deployments. Without it, wheel
   installs default to the per-user data directory
   (`%LOCALAPPDATA%\lunaris\data` on Windows, `$XDG_DATA_HOME/lunaris/data` or
   `~/.local/share/lunaris/data` on Linux); mission data is never written into
   `site-packages`. Verify with `lunaris-data verify --strict --runtime`.
4. **Reproducibility.** Hash-pinned resolutions for Linux + CPython 3.11 live
   in `locks/*.lock.txt`; use them for byte-stable environments
   (`pip install --require-hashes -r locks/requirements-hpc-linux-py311.lock.txt`).

## 4. Proxies

`lunaris-data` uses Python's standard `urllib`, which honors `HTTPS_PROXY` /
`HTTP_PROXY` / `NO_PROXY`. No other component performs network I/O at runtime —
propagation, training, evaluation, and the UIs are fully local.

## 5. Privacy and telemetry

**Lunaris contains no telemetry, crash reporting, analytics, or phone-home of
any kind.** The only network access in the entire product is the explicit
`lunaris-data download` command (official data providers) and whatever your
own `pip` configuration does at install time. All computation results stay on
the machine that produced them.

## 6. Logs and outputs

- Library layers log through Python's `logging` under the `lunaris.*` logger
  namespace; configure handlers/levels through standard `logging` mechanisms
  in your launcher. Progress bars (tqdm) go to stderr.
- Report-producing CLIs (`lunaris-benchmark`, evaluation/validation tools)
  intentionally write their human-readable report to **stdout** — that output
  is the product of the command, not diagnostics. Capture it with shell
  redirection; machine-readable artifacts (JSON/CSV manifests, figures) are
  written next to it under the run's output directory.
- Generated artifacts (runs, benchmarks, figures, checkpoints) go under
  `outputs/` relative to the working directory unless an explicit `--out` is
  given. Batch/HPC output placement is covered in [HPC.md](HPC.md).
- The desktop UIs surface run logs in-app; sessions do not write hidden state
  outside the chosen output/config locations.

## 7. Security boundaries the operator should know

- **`.pt` artifacts are a code-execution boundary.** Loaders are tensor-only
  by default; a legacy artifact that needs full unpickling is refused unless
  explicitly trusted (`--trust-artifact` / `LUNARIS_TRUST_ARTIFACT=1`). Only
  trust artifacts your organization produced or verified. Treat checkpoint
  files from outside like executables.
- **Pre-contract (legacy) surrogate artifacts are refused by default** and
  require `LUNARIS_ALLOW_LEGACY_ARTIFACT=1` (research only, skips strict
  contract validation).
- Data downloads are HTTPS-only with SHA-256 verification against the
  manifest; a digest mismatch aborts the install of that file.
- Report security issues per [SECURITY.md](../SECURITY.md).

## 8. Updating

Pin an exact version in production (`lunaris==0.1.0rc1`). Read
[VERSIONING.md](VERSIONING.md) for the compatibility policy (semver posture,
artifact-schema guarantees, deprecation process) before upgrading, and re-run
`lunaris-data verify --strict --runtime` plus your own acceptance checks after
any upgrade.
