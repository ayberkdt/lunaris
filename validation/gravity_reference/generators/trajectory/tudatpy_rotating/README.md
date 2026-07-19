# Lunaris external validation with TudatPy

This folder is a self-contained runner for the open independent-validation item from the external review: a gravity-only lunar orbit in J2000 under a DE440-rotated MOON_PA JGGRX field, integrated independently by TudatPy and Lunaris.

## What is matched

- Exact JGGRX/GL1800F source file and SHA-256
- Source-file GM and reference radius (SI)
- Fully normalized coefficients, truncated to the scenario's declared
  degree/order (120/120 or 360/360 in the committed evidence)
- J2000 integration frame and MOON_PA coefficient frame
- Exact SPICE kernel files and hashes
- Exact TDB epoch, Cartesian initial state, duration, and output grid
- Gravity-only force set

TudatPy parses the PDS SHADR file with its own small parser and uses classical
fixed-step RK4 at 10 s, 5 s, and 2.5 s. The three levels must demonstrate the
predeclared 3.2--4.8 observed-order interval. Lunaris uses its production
`DynamicsEngine` and SciPy DOP853 with two tolerance/step/orientation-table
settings. A cross-implementation result must satisfy both a predeclared hard
cap and the stricter requirement that it remain below the raw sum of the Tudat
step-convergence and Lunaris tolerance-convergence differences.

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

## Diverse geometry and degree-360 matrix

Five additional contracts exercise geometry, altitude, epoch, and model degree
rather than repeating one initial state. The accepted runs were executed from
clean commits with the exact committed runner bytes; their compact metrics and
complete external-directory hashes are in
[`evidence_matrix_2026_07_19.json`](../../../evidence/tudatpy_rotating/evidence_matrix_2026_07_19.json).

| Contract | Arc | Degree/order | Actual altitude | Actual latitude | Longitude bins | Max SH position difference |
|---|---:|---:|---:|---:|---:|---:|
| polar | 5 d | 120 | 90.2--108.6 km | -90.0--+90.0 deg | 31/36 | 0.831 mm |
| equatorial retrograde | 5 d | 120 | 67.6--131.5 km | -1.16--+1.14 deg | 36/36 | 0.708 mm |
| eccentric low periapsis | 5 d | 120 | 44.1--506.2 km | -89.9--+89.9 deg | 30/36 | 0.677 mm |
| high-altitude retrograde | 5 d | 120 | 447.6--552.5 km | -60.1--+60.1 deg | 36/36 | 0.125 mm |
| high inclination, low altitude | 1 d | 360 | 61.6--78.8 km | -80.3--+80.2 deg | 36/36 | 0.112 mm |

All five passed every rotation, identical-state acceleration, trajectory,
numerical-band, RK4-order, and coverage check. Observed RK4 orders were
4.276--4.737 and maximum identical-state SH acceleration relative differences
were `1.59e-14`--`2.05e-13` against a fixed `5e-10` cap.

An initial exactly-polar one-day degree-360 diagnostic is retained in the
matrix evidence as **FAIL** and excluded from the pass count: all gravity and
numerical checks passed, but one lunar day covered only 7/36 longitude bins
against its predeclared minimum of 10. The replacement 80-degree scenario was
committed with a 30-bin minimum before execution and covered 36/36 bins. No
acceptance threshold was loosened after observing a result.

Generate the five portable contracts deterministically with:

```powershell
.\.venv\Scripts\python.exe "$runner\generate_diverse_scenarios.py" --out "$runner"
```

Run any matrix member by passing its scenario and a distinct external result
directory to `run_all.ps1`, for example:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$runner\run_all.ps1" `
  -Scenario "$runner\scenario_degree360_high_inclination_1day.json" `
  -Results '<external>\results_matrix_degree360_high_inclination_1day' `
  -Mamba '<external>\micromamba.exe' -TudatPrefix '<external>\.tudat-env' `
  -LunarisPython '.venv\Scripts\python.exe'
```

## Scope boundary

This validates point-mass and static tide-free JGGRX gravity with physical
DE440 lunar orientation for the documented contracts. It is strong
cross-implementation evidence, not a proof over every possible state or model
degree. It does not include third-body gravity, solid tides, radiation
pressure, eclipses, relativity, or non-gravitational forces.
