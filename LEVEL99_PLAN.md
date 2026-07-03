# LEVEL 99 PLAN — Lunaris Extension Roadmap (Executable, Value-Filtered Edition)

> Derived from the "LEVEL 99 Fable v" planning memo (2026-07), verified against
> the repository at commit `00f1748` (2026-07-03), and **filtered**: every task
> below states the concrete question it answers or the risk it retires. Items
> from the memo that do not pass that filter are listed in §3 with the reason
> they were cut, so they are not silently re-added later.
>
> Written as self-contained work orders so any competent agent or developer —
> including smaller LLMs with no prior repo context — can execute each task
> without re-deriving the design.

---

## 0. Execution rules (read before ANY task)

1. **One task per branch.** Branch names: `feature/<task-id-kebab>`
   (e.g. `feature/t2-uq-convergence`). Never commit directly to `main`.
2. **Commit authoring:** author commits as the repository user only. Do NOT
   add any `Co-Authored-By:` trailer (including Claude/AI trailers). Ever.
3. **Verification gate — run before declaring any task done:**
   ```
   python -m pytest -q        # full suite green
   ruff check src tests       # lint clean
   lint-imports               # import-linter contracts pass
   ```
4. **Global must-not-touch** (unless a task explicitly says otherwise):
   - `src/lunaris/core/dynamics/` RHS kernels, `src/lunaris/core/propagation/`
   - `src/lunaris/physics/` kernels (SRP, tides, third-body, …)
   - ST-LRPS training/evaluation (`src/lunaris/surrogate/st_lrps/training`, `.../evaluation`)
   - `src/lunaris/batch/backend_policy.py` logic
   If a task seems to require editing these, STOP and report instead.
5. **Repo-specific traps (each has already broken CI or produced wrong results once):**
   - Compat shims must use **dynamic folds** (`__getattr__`-based), never
     static `from x import y` re-exports — `ruff --fix` strips static ones.
   - Moving a function between modules can silently drop its `@njit` decorator — check.
   - `ARCHITECTURE.md` must name every import-linter contract **verbatim** as
     in `pyproject.toml` (`[[tool.importlinter.contracts]]` → `name = "..."`).
     A test enforces this.
   - pyqtgraph OpenGL renders blank offscreen (0×0 framebuffer): 3D views are
     interactive-only with 2D exports; never assert on 3D `grab()`.
   - SHADR gravity files have structural `C00 = 0` — never assume the
     monopole is in the coefficient table.
6. **Banned claims** (code comments, docs, figures, commits, paper text):
   "optimal in seconds", "100% fidelity", "AI pilot", "flight-ready",
   "replaces spherical harmonics", any navigation/OD-accuracy language for UQ
   outputs, any autonomy language for the RL env.
7. **Naming:** use **ensemble** for the statistics layer
   (`lunaris.analysis.ensemble`) and **batch** for the propagation engine
   (`lunaris.batch`). Do not reintroduce `monte_carlo`/`mc_` names
   (canonicalized in commits `06473fc`, `46e61ce`).

---

## 1. Reality delta — what already exists (verified in source; do NOT rebuild)

The memo predates PR #78 and the ensemble rename. Anything marked DONE below
was checked in the code, not assumed.

| Memo item | Status | Where it lives now |
| --- | --- | --- |
| `analysis/monte_carlo/` namespace | RENAMED | `src/lunaris/analysis/ensemble/` |
| UQ report bundler | DONE (#78) | `analysis/ensemble/uq_report.py` — `build_uq_report(...)`; post-hoc CLI `main(argv)` with `--archive --out --survived-only --no-figures` |
| RIC covariance decomposition | DONE (#78) | `statistics.py`: `ric_basis_from_state()`, `compute_ric_uncertainty()`, `RICUncertainty` |
| Linear FD-STM cross-check | DONE (#78) | `analysis/ensemble/linear_check.py` + `propagate_covariance_linear()` |
| CLI wiring | DONE (#78) | `lunaris-batch` runner: `--uq-report-dir` (`cli/batch_runner.py:523`); post-hoc: `python -m lunaris.analysis.ensemble.uq_report` |
| Full (non-diagonal) P₀ sampling | DONE (mechanism) | `StateUncertainty.covariance_6x6` (`common/batch_defs.py:98`) → `cholesky_factor()` → `sample_initial_states()` (`batch/sampling.py:79`) |
| Covariance eigenvalue figure | DONE | `plot_covariance_eigenvalues` (`analysis/ensemble/plotting.py:551`) |
| RIC σ history figure | DONE | `plot_ric_sigma_history` (`plotting.py:525`) |
| Core UQ tests | DONE (#78) | `tests/test_uq_report.py`: PSD/symmetry, zero-dispersion, RIC hand cases, seed-hash reproducibility, manifest completeness, figure smoke, FD-STM linear + two-body |
| UQ doc | DONE (#78) | `docs/UQ_COVARIANCE.md` |
| Tudat reference pattern (gravity) | EXISTS | `validation/independent/tudatpy_reference.py` — SRP variant does NOT exist |
| SRP/eclipse internal validation | EXISTS | `tests/test_srp_eclipse.py`, `docs/FORCE_MODEL_VALIDATION.md` |
| Differentiable propagation | NOT STARTED | planned home: `src/lunaris/optimization/differentiable/` |
| RL environment | NOT STARTED | deferred (see §7) |

---

## 2. The value filter

Every kept task must answer YES to at least one of:

- **(a)** It produces or protects a **paper claim** (figure, table, methods
  sentence, or a reviewer question pre-empted).
- **(b)** It retires a **correctness risk** in something already shipped.
- **(c)** It is a **gate** that cheaply de-risks a larger planned effort
  before money/time is spent on it.

Tasks that only add optionality, symmetry, or "nice to have" outputs are cut
(§3) or deferred (§7).

**The single most valuable item in the entire roadmap is not a new feature:**
it is the Phase-0 paper-evidence run (G0). Everything paper-facing gates on
it. The memo's own conclusion — kept as the plan's spine.

---

## 3. Cut list — memo items removed, with reasons (do not silently re-add)

| Memo item | Why cut |
| --- | --- |
| Eigenvalue-spectrum history figure ("new, trivial") | Already exists (`plot_covariance_eigenvalues`). Re-implementing it is negative-value work. |
| Separate `lunaris-uq` CLI entry point | The post-hoc `python -m lunaris.analysis.ensemble.uq_report` CLI and the `--uq-report-dir` runner flag already cover both workflows. A third entry point adds surface area, docs, and an entry-point-inventory test update for zero new capability. |
| Direct user-supplied ensembles as UQ input | No user or paper figure needs it; the archive-based post-hoc path already decouples propagation from reporting. Revisit only on a concrete request. |
| Parquet / new storage formats | Memo already rejected it; reaffirmed — HDF5/NPZ + audited archives suffice. |
| GMAT SRP cross-check | Script-export + manual runs; Tudat covers the independent-implementation requirement scriptably. GMAT stays optional-forever unless a reviewer demands it. |
| Mahalanobis normality diagnostics as a default report output | `mahalanobis_distance()` exists for ad-hoc use. Shipping a default "normality verdict" invites over-interpretation of a diagnostic; add only if the paper text needs one number. |
| Full RL work-order spec in this plan | RL is deferred until after paper submission (memo §6 reasoning stands). Carrying a detailed executable spec here invites premature starts by exactly the smaller models this plan targets. §7 keeps the gate + hard requirements only. |
| Run-comparison dashboard | Needs a stable comparison schema, which needs ≥2 accepted benchmark runs, which need Phase 0. Nothing to design yet. |
| torchdiffeq/adjoint integration | Explicitly banned until the fixed-step MVP hits a **measured** memory wall. |
| UKF/EnKF/filtering, process noise, panel/attitude SRP, docking/landing envs, multi-agent, web/SaaS | Different projects wearing this project's clothes. Not backlogged. |

---

## 4. Gate G0 — paper-safe evidence run (no new code; maintainer-scheduled)

**Contribution (a):** the first accepted benchmark table; the artifact every
downstream claim cites. Without it, T3's real study and all of Phase D are idle.

1. Run the existing pipeline end to end via `lunaris-st-lrps-paper-evidence`:
   dataset → spatial-block training → validation suite → A0–A6 ablation →
   `--paper-safe` benchmark.
2. Done when: `docs/BENCHMARK_RESULTS.md` gains an accepted table;
   `validation_report.json` passes; `benchmark_manifest.json` has
   `validation.status: passed`.
3. Expensive and state-producing: execute only when explicitly scheduled by
   the maintainer (per the `reproducible-benchmarks` policy). G0 gates *paper
   integration* of everything below, not engineering starts.

---

## 5. Track A — UQ completion (3 tasks, analysis/ only)

### T1 — Full-P₀ provenance in the UQ manifest  (size S; deps none; filter a+b)

**Contribution:** the sampling mechanism for full 6×6 P₀ exists, but if the
manifest doesn't faithfully record the input dispersion in that mode, the
report's provenance story — the whole point of #78 — is silently broken for
exactly the configuration the paper's UQ subsection will use.

**Steps:**
1. Read `build_uq_report()` in `analysis/ensemble/uq_report.py`; locate the
   input-uncertainty serialization.
2. Verify the manifest records: distribution family (`"gaussian_diagonal"` /
   `"gaussian_full"`), the σ vector **or** full 6×6 matrix (nested list,
   canonical-JSON-safe via the existing `_jsonable()`), sampling method,
   seed, N. Add whatever is missing for the `covariance_6x6` case.
3. If the post-hoc archive path cannot recover the input dispersion, write
   `"input_dispersion": "unavailable (post-hoc archive)"` explicitly — the
   manifest must never be silently incomplete.
4. **If step 2 shows everything is already recorded, close the task with the
   two tests below as regression locks and no production change.**

**Tests (`tests/test_uq_report.py`):**
- `test_uq_manifest_records_full_covariance_input` — non-diagonal P₀ (diag +
  one r–v off-diagonal term) → manifest carries the matrix and
  `gaussian_full`.
- `test_full_covariance_sampling_reproduces_input_covariance` — N=4096 via
  `sample_initial_states` with known non-diagonal P₀; `np.cov(...,ddof=1)`
  matches within relative Frobenius < 0.1.

**Acceptance:** both tests green; changes confined to `analysis/ensemble/` + `tests/`.

### T2 — Convergence study: is N sufficient?  (size M; deps none; filter a)

**Contribution:** the first question any reviewer asks about an ensemble
covariance is "how do you know N was enough?". This produces the figure that
answers it (memo paper-figure F3), and it needs no GPU and no trained artifact.

**Files:** `src/lunaris/analysis/ensemble/convergence.py`,
`tests/test_uq_convergence.py`.

**Design (implement exactly this — no CLI beyond the one flag):**
1. ```python
   def run_convergence_study(
       result_or_archive,                 # BatchPropagationResult | str path
       *,
       n_grid: Sequence[int] = (128, 256, 512, 1024, 2048, 4096, 8192),
       epochs: str = "final",             # "final" | "all"
   ) -> ConvergenceStudy
   ```
   For each N: take the **first N members** (prefix subsetting keeps Sobol
   designs valid at power-of-two lengths — see `_sobol_size_note()` in
   `batch/sampling.py`), compute `P_N` through the existing
   `compute_ensemble_statistics` path, record
   `‖P_N − P_ref‖_F / ‖P_ref‖_F` and the max-eigenvalue ratio against
   `P_ref = P at max(n_grid)`.
2. `ConvergenceStudy` dataclass: `n_grid`, `frobenius_rel`, `eig_ratio`,
   `sampling_method`, `seed`; `to_npz(path)` writer.
3. SE proxy at max N: 8 disjoint blocks → per-block covariance → element-wise
   std. Docstring must call it a proxy, not a rigorous estimator.
4. `plot_convergence(studies, out_path)`: log-log Frobenius drift vs N, one
   line per study (enables the MC-vs-Sobol overlay from two archives that
   differ only in `sampling_method`). Matplotlib `Agg`, follow
   `docs/UQ_COVARIANCE.md` figure conventions (units + frame stated).
5. Wire `--convergence` into the existing post-hoc `main()` in
   `uq_report.py`: writes `uq_convergence.npz` + figure into the report dir
   and a summary block into the manifest.

**Tests:**
- `test_convergence_drift_shrinks_for_iid_gaussian` — synthetic i.i.d.
  Gaussian ensemble (no propagation): drift at 8192 < drift at 128 and
  < 0.05 relative.
- `test_convergence_cli_flag_writes_outputs` — tmp_path smoke through
  `main(["--archive",…,"--out",…,"--convergence"])`; assert NPZ keys + figure
  file + manifest block.

**Acceptance:** runs on an existing archive without re-propagation; the
MC-vs-Sobol figure produced once from two real archives and added to
`docs/UQ_COVARIANCE.md`; suite green.

### T3 — Backend comparison: SH vs ST-LRPS covariance evolution  (size M; deps: plumbing none / study G0; filter a+c)

**Contribution:** converts the paper's throughput claim into a validated
capability ("the cheap GPU ensemble gives the *same covariance* as the SH
reference") and doubles as orbit-level surrogate validation. Claim only
consistency for the tested configuration.

**Files:** `src/lunaris/analysis/ensemble/backend_comparison.py`,
`tests/test_uq_backend_comparison.py`.

**Steps:**
1. `compare_backends(config, backend_a, backend_b, *, seed, n_samples) -> BackendComparison`:
   build ONE standard-normal design via `generate_standard_normal_design`
   (same seed); map through the same `StateUncertainty`; run the batch engine
   twice with the two **requested** backends; after each run assert
   `actual == requested` using the existing backend-policy provenance and
   **raise with a clear message on mismatch** — never compare a silent
   fallback.
2. Metrics per epoch: `frobenius_rel_diff(P_a, P_b)`, RIC σ differences,
   mean-state RIC offset. Output: `backend_comparison.npz` + two-panel figure
   (RIC σ both backends; relative Frobenius diff vs time) + manifest block
   naming both backends, their provenance hashes, and the surrogate artifact
   hash.
3. No new console entry point: expose via
   `python -m lunaris.analysis.ensemble.backend_comparison`
   (`--config --backend-a --backend-b --n --seed --out`).

**Tests (CPU-only, artifact-free — this is the cheap de-risk):**
- `test_backend_self_comparison_is_zero` — a backend vs itself: all metrics
  ≈ 0 to round-off. Validates plumbing everywhere, including CI.
- `test_backend_comparison_mismatch_guard` — request an unavailable backend;
  assert the informative raise (no silent fallback into the comparison).

**The real SH-vs-surrogate study is not a unit test:** run once after G0,
append figure + numbers to `docs/UQ_COVARIANCE.md` with consistency-only
wording.

---

## 6. Track B — SRP external validation (tests + one reference module; zero kernel edits)

**Phase rule:** validation work, not features. Expected kernel edits: **zero**.
If a comparison exposes a real discrepancy: stop, open an issue, fix it as a
separate documented bugfix + regression test. Never tune a kernel inside a
validation PR to make a comparison pass.

**Why this track earns its place (filter a):** the paper's methods section
says the force models were validated; today every SRP check is
same-author/same-code. One independent-path tier (T4) plus one cross-tool tier
(T5) is the difference between "we tested our code against itself" and a
defensible sentence. The matrix in the memo shows the gap is exactly one
column: external cross-validation.

### T4 — Independent-path shadow & limiting-case tests  (size M; deps none; filter a+b)

One PR, three test files, production untouched.

**(i) `tests/test_srp_shadow_geometry.py`**
1. Write a small **independent** conical-shadow geometry helper *inside the
   test file* (different author path — never call the production
   `_shadow_factor_conical` to generate its own expectations). Geometry: with
   Sun radius R_s, occulting-body radius R_b, Sun–body distance d, the umbra
   cone half-angle is `asin((R_s − R_b)/d)`, penumbra `asin((R_s + R_b)/d)`;
   derive boundary offsets at 2–3 fixed geometries (derivation in comments).
2. Assert production shadow factor = 0 on the umbra inner edge, 1 on the
   penumbra outer edge, strictly in (0,1) between.
3. β-angle sweep on a circular orbit: eclipse fraction per orbit monotonically
   decreasing in β and exactly zero above the analytic no-eclipse β.

**(ii) `tests/test_srp_shadow_timing.py`** — the genuinely missing coverage:
current tests check shadow *values*, not *event epochs*. Propagate a short arc
through an eclipse; bisect the production shadow factor on the dense output
for entry/exit epochs; compare to the independent helper with a stated Δt
tolerance (justify it from dense-output accuracy in a comment).

**(iii) `tests/test_srp_limits.py`**
- `test_zero_cr_a_over_m_recovers_gravity_only` — Cr·A/m → 0: SRP-enabled
  trajectory matches gravity-only to integrator tolerance over ≥ 1 orbit
  (reuse the vector-atol convention from
  `validation/independent/cross_validation.py`).
- `test_deep_umbra_srp_acceleration_and_work_are_zero` — arc fully inside
  umbra: SRP acceleration exactly 0 throughout; SRP work integral 0 to
  round-off.

**Acceptance:** all green in the normal suite (no external deps); zero
production diffs.

### T5 — Tudat SRP cross-check  (size M; deps none; external friction expected; filter a)

**File:** `validation/independent/tudatpy_srp_reference.py`, mirroring the
existing `tudatpy_reference.py` (gravity) structure exactly: same `outputs/`
conventions, same provenance capture.

1. **Accelerations at fixed geometries first** (≥100 samples spanning
   sunlit/penumbra/umbra), only then a short SRP-enabled trajectory diff —
   this separates model-definition mismatches from integration differences.
2. Every model-definition choice that must match goes in the module docstring
   *before* running: solar-flux constant, AU value, occulting-body radii,
   Tudat shadow-model type. Mismatches here masquerade as bugs.
3. Guard with a `requires_tudat` pytest marker (add to `conftest.py` mirroring
   the existing optional-dependency markers); must skip cleanly on Windows —
   the study runs on the Linux/HPC side.
4. **Go/no-go:** if Tudat friction stalls this > 2 weeks, ship T4 alone and
   defer cross-tool. T4 already upgrades the methods sentence.

### T6 — Documentation close-out  (size XS; deps T4[, T5]; filter a)

1. Append results (tolerances, sample counts, versions) to
   `docs/FORCE_MODEL_VALIDATION.md`. Keep the existing "engineering
   approximation" wording — do not strengthen claims.
2. **Conditional, not preemptive:** the benchmark-manifest SRP block
   (`spacecraft {mass_kg, area_m2, cr}`, `srp {model, shadow, bodies}`,
   ephemeris source+hash, enabled-forces list) is added only if/when a
   *perturbed* benchmark is actually published. Gravity-only manifests stay
   untouched.

---

## 7. Track C — differentiable propagation (research bet; explicit kill criterion)

**Status:** paper future-work only (one sentence, memo §8 wording verbatim).
**Start gate:** after G0 exists (needs a trained `potential_autograd`
artifact) and Track A tasks T1–T2 are merged. **Filter (c):** the cheap steps
(C1–C2) de-risk the expensive ones; the kill criterion (C5) caps the downside.

**Non-negotiable architecture (decided; do not revisit):**
- Home `src/lunaris/optimization/differentiable/` — NOT under
  `surrogate/st_lrps/` (consumes the runtime artifact through its public
  contract; keeps a future torch-SH baseline surrogate-agnostic; keeps the
  "ST-LRPS runtime (inference path) stays light" contract intact).
- **No retrofits:** `core.propagation` (scipy/Numba) and
  `torch_batch_propagator` (six `.item()/.numpy()/no_grad/detach` sites,
  inference-tuned) stay untouched. Import only
  `surrogate.st_lrps.runtime.force_model` + `shared`/`common`.
- Out of the differentiable path: NumPy round-trips, Numba, runtime SPICE
  (precompute ephemeris/rotation tables as constant tensors), hard eclipse
  switches (gravity-only MVP), `.item()` in the loop, impact events (fixed
  horizon only).

**C1 — skeleton + import contract (S).** Create the package; add
`[[tool.importlinter.contracts]]`
`name = "optimization imports only the ST-LRPS runtime, never training/evaluation/UI"`
(forbidden: `lunaris.optimization` → `...st_lrps.training`, `...evaluation`,
`lunaris.ui`, `lunaris.cli`); copy the name **verbatim** into
`ARCHITECTURE.md`; lazy torch imports per the runtime package's pattern.

**C2 — torch RK4 + point-mass parity gate (M).**
`rk4_rollout(f, y0, t_grid) -> (T,B,6)` — fixed-step, float64, batch-first,
no `.item()`, no autograd-breaking in-place ops. `point_mass_accel(r, mu)` in
torch. Gate test `tests/test_diffprop_parity.py::test_zero_dv_point_mass_parity_vs_cpu`:
zero-ΔV rollout vs `core.propagation` on a 2-orbit near-circular LLO arc,
match within integrator tolerance. (Kepler-reference trap: the classical
branch in `validation/independent/cross_validation.py` fails on eccentric
orbits — use the UV formulation or a near-circular case.)

**C3 — surrogate-in-the-loop (M; needs G0 artifact).** Wrap
`potential_autograd` + scalers as `surrogate_accel`; body-fixed↔inertial via
**precomputed per-step rotation matrices** (constant tensors). Gate: zero-ΔV
surrogate rollout vs the existing non-differentiable batch backend with the
same artifact, ≤1 orbit, stated tolerance. This is the frame-error gate — do
not proceed until it passes.

**C4 — single-impulse targeting + gradient validation (M).** One ΔV∈ℝ³ at t₀;
terminal-miss loss over ≤1 orbit; Adam and L-BFGS. Tests
(`tests/test_diffprop_gradients.py`, CPU-only where possible): central-FD vs
autograd per component (~1e-6 relative, float64, point-mass); strict loss
decrease for k steps; analytic anchor — residual disabled ⇒ optimized ΔV
matches the closed-form two-body answer. Horizon-vs-gradient-quality study is
a memo exhibit, not a unit test. All solutions reported as **local**.

**C5 — pre-registered kill criterion (write BEFORE running).** If autograd
does not beat an FD-based optimizer baseline (FD + Nelder-Mead/SLSQP) on
wall-clock **or** robustness for the MVP problem, write up the negative
result and stop the track. Gradient checkpointing / adjoint methods only
after a **measured** memory wall — never preemptively.

---

## 8. Deferred — gates only, no work orders

- **RL environment** (`lunaris/envs/`, `[rl]` extra): do not start before the
  paper is submitted. When started, hard requirements from memo §6 apply:
  zero-action rollout bit-identical to plain `propagate()`; deadband
  station-keeping baseline mandatory before any RL claim; RIC observation
  reuses the single `analysis/ensemble/statistics.py` convention; gymnasium
  never enters core deps; framing is "research/demo simulation environment",
  station-keeping only. The detailed design lives in the source memo — do not
  re-derive it here until the gate opens.
- **UQ Results-zone UI panel:** fold into the existing UI roadmap P1a
  (Results-zone figure gallery) rather than running as a separate effort —
  the UQ report already writes PNG figures + manifest into a run directory,
  which is exactly what P1a displays. No standalone task.
- **Run-comparison dashboard / artifact inspector UI:** wait for ≥2 accepted
  benchmark runs (schema stability), i.e. after G0 has run at least twice.

---

## 9. Execution queue (small-model order, cheapest de-risk first)

| # | Task | Needs GPU/artifact? | Rationale |
| --- | --- | --- | --- |
| 1 | T1 | no | S-size; locks manifest honesty before studies rely on it |
| 2 | T4 | no | tests-only, zero production risk, upgrades methods sentence |
| 3 | T2 | no | the "was N enough?" paper figure; pure analysis |
| 4 | T3 plumbing + self-comparison tests | no | de-risks the study for free |
| 5 | T5 | no (Linux/HPC) | external-friction track, run in parallel; 2-week go/no-go |
| 6 | **G0** (maintainer-scheduled) | yes | gates all paper integration |
| 7 | T3 real study; T6 | yes / no | consume G0 artifact and T4–T5 results |
| 8 | C1 → C5 | C3+ yes | research bet, kill criterion armed |
| 9 | §8 items | — | only after their gates open |

**Definition of done for the whole plan:** the paper carries (F-UQ) one
ellipsoid/RIC figure + (F-conv) the convergence figure + (T-prov) the
provenance table, backed by a G0-accepted benchmark; `FORCE_MODEL_VALIDATION.md`
cites at least the T4 independent-path tier; Track C has either a working
prototype memo or an honest negative-result memo. Everything else is
explicitly deferred, not silently pending.
