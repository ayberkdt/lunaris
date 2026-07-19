# Lunaris external validation with TudatPy

This folder is a self-contained runner for the open independent-validation item from the external review: a gravity-only lunar orbit in J2000 under a DE440-rotated MOON_PA JGGRX field, integrated independently by TudatPy and Lunaris.

## What is matched

- Exact JGGRX/GL1800F source file and SHA-256
- Source-file GM and reference radius (SI)
- Fully normalized coefficients, truncated to degree/order 120/120
- J2000 integration frame and MOON_PA coefficient frame
- Exact SPICE kernel files and hashes
- Exact TDB epoch, Cartesian initial state, duration, and output grid
- Gravity-only force set

TudatPy parses the PDS SHADR file with its own small parser and uses classical fixed-step RK4 at 5 s and 2.5 s. Lunaris uses its production `DynamicsEngine` and SciPy DOP853 with two tolerance/step/orientation-table settings. This provides both implementation and integrator independence plus within-tool convergence controls.

## Run

Create the pinned Tudat environment in an external scratch directory; do not
place conda environments or downloaded package caches in the repository. Then
run from PowerShell, passing the external micromamba executable/environment and
the repository virtual-environment Python explicitly:

```powershell
$runner = 'validation\gravity_reference\generators\trajectory\tudatpy_rotating'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$runner\run_all.ps1" `
  -Mamba '<external>\micromamba.exe' `
  -TudatPrefix '<external>\.tudat-env' `
  -LunarisPython '.venv\Scripts\python.exe'
```

To keep long-arc artifacts separate:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$runner\run_all.ps1" `
  -Scenario "$runner\scenario_5day.json" -Results 'outputs\validation\tudat\results_5day' `
  -Mamba '<external>\micromamba.exe' -TudatPrefix '<external>\.tudat-env' `
  -LunarisPython '.venv\Scripts\python.exe'

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$runner\run_all.ps1" `
  -Scenario "$runner\scenario_30day.json" -Results 'outputs\validation\tudat\results_30day' `
  -Mamba '<external>\micromamba.exe' -TudatPrefix '<external>\.tudat-env' `
  -LunarisPython '.venv\Scripts\python.exe'
```

The command exits nonzero if any acceptance check fails. Results and provenance
are written under the selected `-Results` directory; start with
`VALIDATION_REPORT.md` and `comparison_summary.json` there.

The environment is pinned in `environment.yml`. The evidence run used
micromamba 2.8.1 only as a conda-compatible package manager; TudatPy itself was
the official `tudat-team::tudatpy` 1.0.0 Windows build.

If the local environment ever needs to be rebuilt, run:

```powershell
& '<external>\micromamba.exe' create --yes `
  --root-prefix '<external>\.mamba-root' `
  --prefix '<external>\.tudat-env' `
  --strict-channel-priority --file "$runner\environment.yml"
```

## Longer arcs

The committed `scenario.json`, `scenario_5day.json`, and
`scenario_30day.json` contracts cover the qualified one-, five-, and thirty-day
arcs. Evidence produced on 2026-07-19 is summarized and checksummed in
[`evidence_2026_07_19.json`](../../../evidence/tudatpy_rotating/evidence_2026_07_19.json).
Full CSV histories remain in external scratch and are intentionally not
committed.

## Scope boundary

This validates point-mass and static tide-free JGGRX gravity with physical DE440 lunar orientation. It does not include third-body gravity, solid tides, radiation pressure, eclipses, relativity, or non-gravitational forces.
