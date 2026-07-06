# Uncertainty Quantification: Ensemble Covariance & Error Ellipsoids

This page documents the ensemble uncertainty-quantification (UQ) report:
what the propagated covariance *is*, how to produce a provenance-stamped UQ
run, what the outputs contain, and how the result is validated.

**Terminology.** The batch workflow that perturbs injection conditions and
spacecraft parameters is the **Injection Dispersion Analysis** (Enjeksiyon
Tolerans Analizi). The **UQ report** described here is the covariance
*evidence product* generated from an ensemble archive of such a run. Neither
is orbit determination, navigation covariance, or process-noise estimation;
"force-model uncertainty" (perturbation budget) and numerical solver
"tolerances" remain separate concepts.

## What the covariance is (and is not)

> The reported covariance is the **unbiased sample covariance** (`ddof=1`) of
> the ensemble state in the **Moon-centred inertial integration frame** at the
> shared output epochs, induced by the **declared initial-state and
> spacecraft-parameter dispersion** propagated through **deterministic
> dynamics**.

It contains **no process noise** and **no measurement updates**. It is a
forward-propagated dispersion analysis, **not** an orbit-determination
covariance, and it must never be presented as navigation performance. This
definition is embedded verbatim in every `uq_manifest.json`.

Initial uncertainty is currently a diagonal Gaussian (`StateUncertainty` /
`SpacecraftUncertainty` per-component σ) mapped through a standard-normal
design; the design can be classical Monte Carlo, Latin Hypercube, or (scrambled)
Sobol (`--sampling-method`). The distribution family, σ values, method, seed,
and sample count are all recorded in the manifest.

## Producing a UQ report

With a fresh batch/ensemble run (all `lunaris-batch` physics/backend flags
apply; `lunaris-mc` is retained as a historical command name for the same
runner):

```bash
lunaris-batch --n-samples 512 --seed 42 --sampling-method sobol_scrambled \
    --alt-km 100 --inc-deg 60 --days 1 \
    --mc-output-path outputs/ensemble/llo_uq.h5 \
    --uq-report-dir outputs/ensemble/llo_uq_report
```

Post-hoc, from an existing archive (no re-propagation):

```bash
python -m lunaris.analysis.ensemble.uq_report \
    --archive outputs/ensemble/llo_uq.h5 \
    --out outputs/ensemble/llo_uq_report
```

A failed UQ report fails the CLI run (exit code 3): an explicitly requested
evidence product is never silently skipped.

## Report contents

| File | Contents |
|---|---|
| `uq_covariance.npz` | `t_s (T,)`, `mean_state (T,6)`, `cov (T,6,6)`, `cov_ric (T,3,3)`, `sigma_ric_m (T,3)`, `ellipsoid_semi_axes_3sigma_m (T,3)`, `ellipsoid_eigvecs (T,3,3)`, `alt_mean_km`, `alt_std_km` |
| `uq_summary.csv` | per-epoch scalars: total position/velocity 1-σ, RIC 1-σ components, max 3-σ ellipsoid semi-axis, altitude mean/σ |
| `uq_manifest.json` | canonical JSON: covariance definition, ensemble counts, run-config echo + hash, source-archive path + SHA-256, archive metadata, per-file SHA-256, **covariance content hash**, git commit/dirty state, environment |
| `figures/` | σ/correlation history, RIC 1-σ history, covariance eigenvalue spectrum, 3-D ensemble with 3-σ ellipsoids, altitude envelope |

The **covariance content hash** is a SHA-256 over the numerical arrays
themselves (names, shapes, raw float64 bytes), independent of file timestamps:
re-running with the same seed and configuration must reproduce the identical
hash. This is the reproducibility acceptance criterion, enforced by
`tests/test_uq_report.py`.

RIC components use the same convention as the orbit-benchmark error
decomposition: **R** = radial (r̂), **C** = cross-track (orbit normal,
(r×v)/|r×v|), **I** = in-track/along-track (C×R); a test pins the analysis-layer
projection to the benchmark implementation so the two cannot drift.

## Validation

- **Symmetry/PSD** — every `P(t)` is checked symmetric and positive
  semi-definite (to round-off) in the test suite.
- **Zero-dispersion degenerate case** — zero input σ yields zero covariance.
- **Seed reproducibility** — identical seeds produce identical content hashes;
  different seeds differ.
- **Linear (STM) cross-check** — `lunaris.analysis.ensemble.linear_check`
  builds state-transition matrices by central finite differences of any
  propagation callable and compares `Φ P₀ Φᵀ` with the ensemble covariance.
  For exactly linear dynamics the two agree to sampling error; for a short
  point-mass arc with small dispersion they agree within the sampling +
  mild-nonlinearity budget (both are locked as tests). On real force models,
  the epoch where MC and linear histories diverge is itself a result — the
  onset of non-linearity for that dispersion — not a failure.

## Claim discipline

Do **not** claim from these outputs: navigation or orbit-determination
accuracy; Gaussianity of the propagated distribution (the ensemble is Gaussian
at t₀ only); validity outside the sampled dispersion magnitudes, scenario, and
force-model configuration; anything about runs whose manifest is missing or
whose content hash does not reproduce.

Statistics come from `lunaris.analysis.ensemble.statistics`
(mean/covariance tube, 3-σ ellipsoids, RIC uncertainty, impact statistics with
Wilson CIs, orbital-element dispersion); figures from
`lunaris.analysis.ensemble.plotting`.
