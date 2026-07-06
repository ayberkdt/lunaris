# LEVEL 99 PLAN — Lunaris Extension Roadmap (Verified, Value-Filtered Edition)

> Derived from the "LEVEL 99 Fable v" planning memo (2026-07). Every factual
> premise below was **verified in source** at commit `00f1748` (2026-07-03) —
> nothing is assumed from the memo. Tasks that turned out to be already done,
> unreachable, or duplicative were cut and are listed in §3 with evidence, so
> they are not silently re-added later.
>
> Written as self-contained work orders so any competent agent or developer —
> including smaller LLMs with no prior repo context — can execute each task
> without re-deriving the design.

---

## 0. Execution rules (read before ANY task)

1. **One task per branch.** Branch names: `feature/<task-id-kebab>`
   (e.g. `feature/a1-uq-convergence`). Never commit directly to `main`.
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
   outputs, any autonomy language for the RL env. `gpu_st_lrps_direct`
   (force_direct) results back **no claims anywhere** pending curl/orbit-level
   validation.
7. **Naming:** **ensemble** for the statistics layer
   (`lunaris.analysis.ensemble`), **batch** for the propagation engine
   (`lunaris.batch`). Do not reintroduce `monte_carlo`/`mc_` names
   (canonicalized in commits `06473fc`, `46e61ce`).

---

## 1. Verified ground truth (checked in source — the plan's factual basis)

| Fact | Evidence |
| --- | --- |
| UQ report bundler, RIC decomposition, FD-STM linear check, post-hoc CLI, core tests, `docs/UQ_COVARIANCE.md` — all DONE | PR #78: `analysis/ensemble/uq_report.py` (`build_uq_report`, `main` with `--archive --out --survived-only --no-figures`), `statistics.py:575–711`, `linear_check.py`, `tests/test_uq_report.py` |
| The UQ manifest already records the full batch config verbatim + hash | `cli/batch_runner.py:532` passes `run_config=asdict(batch_cfg)`; `uq_report.py:323–326` stores it + `canonical_json_sha256`. Post-hoc path stores explicit `run_config: null` (not silently incomplete) |
| Batch archives store the **full per-member trajectory tensor** | `batch/storage.py`: dataset `Y` shape `(T, n_samples, 6)`, HDF5 with lazy loading (`load_batch_result(..., lazy=True)`) — post-hoc studies need no re-propagation |
| Full non-diagonal P₀ is mechanism-only, **unreachable from any config/CLI** | `StateUncertainty.covariance_6x6` exists (`common/batch_defs.py:98`) and `sample_initial_states` applies it via `cholesky_factor()`, but `covariance_6x6` appears in no other file under `src/lunaris` |
| SH batch backends | `cpu_sh`, `numba_cuda_sh`, `torch_cuda_sh`, `torch_cpu_sh` (`batch/backend_policy.py:171`), with requested-vs-actual + fallback-provenance machinery |
| Surrogate batch backends | `cpu_st_lrps`, `gpu_st_lrps_potential`, `gpu_st_lrps_direct` (`batch/engine.py:54–59`), model dir validated via `validate_st_lrps_model_dir` |
| Covariance eigenvalue + RIC σ figures already exist | `plot_covariance_eigenvalues` (`plotting.py:551`), `plot_ric_sigma_history` (`plotting.py:525`) |
| SRP internal tests already cover values | `tests/test_srp_eclipse.py`: closed-form magnitude, direction, inverse-square, shadow bounds, day-side, **deep-umbra zero force**, **penumbra continuity/monotonicity**, zero mass/area, eclipse-disable, wrapper parity |
| Tudat reference pattern exists for gravity only | `validation/independent/tudatpy_reference.py`; no SRP variant |
| Differentiable propagation / RL | Not started anywhere |

---

## 2. The value filter

Every kept task must answer YES to at least one of:

- **(a)** produces or protects a **paper claim** (figure, table, methods
  sentence, reviewer question pre-empted);
- **(b)** retires a **correctness risk** in something already shipped;
- **(c)** is a cheap **gate** that de-risks a larger planned effort before
  time is spent on it.

**The single most valuable item in the roadmap is not a new feature:** it is
the Phase-0 paper-evidence run (G0). Everything paper-facing gates on it.

The surviving backlog is deliberately small: **two UQ tasks, two SRP tasks,
one doc close-out, one gated research track.**

---

## 3. Cut list — removed items, with evidence (do not silently re-add)

| Item | Why cut |
| --- | --- |
| "Record full-P₀/dispersion in the UQ manifest" | Already covered: the runner passes `asdict(batch_cfg)` verbatim into the manifest with a canonical hash (`batch_runner.py:532`, `uq_report.py:323`). Nothing to add. |
| Wiring `covariance_6x6` into config/CLI | The mechanism is dead code today (§1), but the current paper's UQ subsection uses diagonal σ — the memo itself called full-P₀ a fast-follow. Wire it only when a concrete scenario needs a correlated P₀; until then it is untestable surface area. |
| Eigenvalue-spectrum history figure | Already exists: `plot_covariance_eigenvalues` (`plotting.py:551`). |
| Deep-umbra zero-SRP test; penumbra continuity test | Already exist in `tests/test_srp_eclipse.py` (`test_deep_umbra_zeros_the_force`, `test_penumbra_is_continuous_and_monotone`). My earlier draft duplicated them. |
| Deep-umbra SRP **work-integral** test | Pointwise-zero acceleration is already tested; the integral of an identically-zero function adds no information. |
| β-angle eclipse-fraction sweep | Weak marginal value over the entry/exit **epoch** test (B1), which already validates orbit-level shadow geometry against an independent implementation. |
| Separate `lunaris-uq` CLI entry point | Post-hoc `python -m lunaris.analysis.ensemble.uq_report` + the runner's `--uq-report-dir` cover both workflows; a third entry point is surface area for zero capability. |
| Direct user-supplied ensembles as UQ input | No user or figure needs it; the archive-based post-hoc path already decouples propagation from reporting. |
| Parquet / new storage formats | HDF5/NPZ + audited archives suffice (memo agreed; reaffirmed). |
| GMAT SRP cross-check | Script-export + manual runs; Tudat satisfies the independent-implementation requirement scriptably. |
| Mahalanobis "normality verdict" in the default report | `mahalanobis_distance()` exists for ad-hoc use; a default verdict invites over-interpretation. |
| Full RL work-order spec in this plan | RL is deferred until after paper submission; carrying an executable spec here invites premature starts. §7 keeps the gate + hard requirements only. |
| Run-comparison dashboard | Needs a stable schema → needs ≥2 accepted benchmark runs → needs G0 twice. Nothing to design yet. |
| torchdiffeq/adjoint | Banned until the fixed-step MVP hits a **measured** memory wall. |
| UKF/EnKF/filtering, process noise, panel/attitude SRP, docking/landing envs, multi-agent, web/SaaS | Different projects wearing this project's clothes. Not backlogged. |

---

## 4. Gate G0 — paper-safe evidence run (no new code; maintainer-scheduled)

**Contribution (a):** the first accepted benchmark table; the artifact every
downstream claim cites.

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

## 5. Track A — UQ completion (2 tasks, `analysis/` + `tests/` only)

### A1 — Convergence study: "was N enough?"  (size M; deps none; filter a)

**Contribution:** the first question a reviewer asks about an ensemble
covariance. Produces the memo's paper figure F3. Needs no GPU, no artifact,
no re-propagation (archives carry full `Y` — §1).

**Files:** `src/lunaris/analysis/ensemble/convergence.py`,
`tests/test_uq_convergence.py`.

**Design (implement exactly this):**
1. ```python
   def run_convergence_study(
       result_or_archive,                 # BatchPropagationResult | str path
       *,
       n_grid: Sequence[int] = (128, 256, 512, 1024, 2048, 4096, 8192),
       epochs: str = "final",             # "final" | "all"
   ) -> ConvergenceStudy
   ```
   Load via `load_batch_result(path, lazy=True, strict=False)` when given a
   path. For each N (clipped to `n_samples`; skip N > available with a note):
   slice the member axis — `Y_N = Y[:, :N, :]` — into a trimmed shallow copy
   of the result (member axis is axis 1; **prefix** subsetting keeps Sobol
   designs valid at power-of-two lengths, see `_sobol_size_note()` in
   `batch/sampling.py`), reuse `compute_ensemble_statistics`, and record
   `‖P_N − P_ref‖_F / ‖P_ref‖_F` plus the max-eigenvalue ratio against
   `P_ref = P at max usable N`.
2. `ConvergenceStudy` dataclass: `n_grid`, `frobenius_rel`, `eig_ratio`,
   `sampling_method`, `seed`; `to_npz(path)` writer.
3. SE proxy at max N: 8 disjoint member blocks → per-block covariance →
   element-wise std. Docstring must call it a proxy, not a rigorous estimator.
4. `plot_convergence(studies, out_path)`: log-log Frobenius drift vs N, one
   line per study (enables the MC-vs-Sobol overlay from two archives that
   differ only in `sampling_method`). Matplotlib `Agg`; follow
   `docs/UQ_COVARIANCE.md` figure conventions (units + frame stated).
5. Wire `--convergence` into the existing post-hoc `main()` in
   `uq_report.py`: writes `uq_convergence.npz` + figure into the report dir
   and a summary block into the manifest.

**Tests:**
- `test_convergence_drift_shrinks_for_iid_gaussian` — synthetic i.i.d.
  Gaussian ensemble (no propagation): drift at 8192 < drift at 128 and
  < 0.05 relative.
- `test_convergence_cli_flag_writes_outputs` — tmp_path smoke through
  `main(["--archive",…,"--out",…,"--convergence"])`; assert NPZ keys, figure
  file, manifest block.

**Acceptance:** runs on an existing archive without re-propagation; the
MC-vs-Sobol figure produced once from two real archives and added to
`docs/UQ_COVARIANCE.md`; suite green.

### A2 — Backend comparison: `cpu_sh` vs `gpu_st_lrps_potential`  (size M; plumbing deps none, study deps G0; filter a+c)

**Contribution:** converts the paper's throughput claim into a validated
capability ("the cheap GPU ensemble gives the *same covariance* as the SH
reference") and doubles as orbit-level surrogate validation. Claim only
consistency for the tested configuration. **Never** use
`gpu_st_lrps_direct` here (rule §0.6).

**Files:** `src/lunaris/analysis/ensemble/backend_comparison.py`,
`tests/test_uq_backend_comparison.py`.

**Steps:**
1. `compare_backends(cfg, batch_cfg, backend_a, backend_b, *, st_lrps_model_dir=None) -> BackendComparison`:
   build ONE standard-normal design via `generate_standard_normal_design`
   (same seed) and pass it to both runs so the input ensemble is identical;
   run `BatchPropagationEngine` twice with configs differing **only** in the
   backend field (+ `st_lrps_model_dir` for the surrogate side, validated by
   the existing `validate_st_lrps_model_dir`); after each run compare the
   **actual** backend recorded in the result's provenance/diagnostics against
   the requested one and **raise with a clear message on mismatch** — never
   compare a silent fallback.
2. Metrics per epoch: `frobenius_rel_diff(P_a, P_b)`, RIC σ differences,
   mean-state RIC offset. Output: `backend_comparison.npz` + two-panel figure
   (RIC σ both backends; relative Frobenius diff vs time) + a manifest block
   naming both backends, their provenance hashes, and the surrogate artifact
   hash.
3. No new console entry point: expose via
   `python -m lunaris.analysis.ensemble.backend_comparison`
   (`--config --backend-a --backend-b --n --seed --out`).

**Tests (CPU-only, artifact-free — the cheap de-risk):**
- `test_backend_self_comparison_is_zero` — `cpu_sh` vs `cpu_sh`, small N,
  short arc: all difference metrics ≈ 0 to round-off. Validates plumbing in
  CI without GPU.
- `test_backend_comparison_mismatch_guard` — request an unavailable backend;
  assert an informative raise (no silent fallback into the comparison).

**The real SH-vs-surrogate study is not a unit test:** run once after G0 with
`gpu_st_lrps_potential`; append figure + numbers to `docs/UQ_COVARIANCE.md`
with consistency-only wording.

---

## 6. Track B — SRP external validation (tests + one reference module; zero kernel edits)

**Phase rule:** validation work, not features. Expected kernel edits: **zero**.
If a comparison exposes a real discrepancy: stop, open an issue, fix as a
separate documented bugfix + regression test. Never tune a kernel inside a
validation PR.

**Why it earns its place (a):** every existing SRP check is
same-author/same-code (§1 — values are well covered internally). The genuine
gaps are: no independent-path check of the shadow **geometry**, no **event
epoch** (timing) coverage, no trajectory-level limiting case, and no
cross-tool comparison. B1 closes the first three cheaply; B2 closes the last.

### B1 — Independent-path geometry, timing, and limiting-case tests  (size M; deps none; filter a+b)

One PR, two test files, production untouched. **Do not duplicate existing
coverage** in `tests/test_srp_eclipse.py` (§1 lists it).

**(i) `tests/test_srp_shadow_geometry_timing.py`**
1. Write a small **independent** conical-shadow geometry helper *inside the
   test file* (different author path — never call the production
   `_shadow_factor_conical` to generate its own expectations). Geometry: with
   Sun radius R_s, occulting-body radius R_b, Sun–body distance d: umbra cone
   half-angle `asin((R_s − R_b)/d)`, penumbra `asin((R_s + R_b)/d)`; derive
   boundary offsets at 2–3 fixed geometries, derivation in comments.
2. Boundary placement: production shadow factor = 0 on the umbra inner edge,
   = 1 on the penumbra outer edge, strictly in (0,1) between.
3. Entry/exit **epochs** (the missing coverage class — values are tested,
   timing is not): propagate a short arc through an eclipse; bisect the
   production shadow factor on the dense output for entry/exit epochs;
   compare to the independent helper with a stated Δt tolerance, justified
   from dense-output accuracy in a comment.

**(ii) `tests/test_srp_limits.py`**
- `test_zero_cr_a_over_m_recovers_gravity_only` — trajectory-level wiring
  check (pointwise zero is already covered): SRP enabled with Cr·A/m → 0
  matches the gravity-only trajectory to integrator tolerance over ≥ 1 orbit
  (reuse the vector-atol convention from
  `validation/independent/cross_validation.py`).

**Acceptance:** green in the normal suite (no external deps); zero production
diffs.

### B2 — Tudat SRP cross-check  (size M; deps none; external friction expected; filter a)

**File:** `validation/independent/tudatpy_srp_reference.py`, mirroring
`tudatpy_reference.py` (gravity) exactly: same `outputs/` conventions, same
provenance capture.

1. **Accelerations at fixed geometries first** (≥100 samples spanning
   sunlit/penumbra/umbra), only then a short SRP-enabled trajectory diff —
   separates model-definition mismatches from integration differences.
2. Every model-definition choice that must match goes in the module docstring
   *before* running: solar-flux constant, AU value, occulting-body radii,
   Tudat shadow-model type. Mismatches here masquerade as bugs.
3. Guard with a `requires_tudat` pytest marker (add to `conftest.py`
   mirroring existing optional-dependency markers); must skip cleanly on
   Windows — the study runs on the Linux/HPC side.
4. **Go/no-go:** if Tudat friction stalls this > 2 weeks, ship B1 alone and
   defer cross-tool. B1 already upgrades the methods sentence from "tested
   against itself" to "checked against an independent implementation".

### B3 — Documentation close-out  (size XS; deps B1[, B2]; filter a)

1. Append results (tolerances, sample counts, versions) to
   `docs/FORCE_MODEL_VALIDATION.md`. Keep the existing "engineering
   approximation" wording — do not strengthen claims.
2. **Conditional, not preemptive:** the benchmark-manifest SRP block
   (`spacecraft {mass_kg, area_m2, cr}`, `srp {model, shadow, bodies}`,
   ephemeris source+hash, enabled-forces list) is added only if/when a
   *perturbed* benchmark is actually published. Gravity-only manifests stay
   untouched.

---

## 7. Track C — differentiable propagation (research bet; kill criterion armed)

**Status:** paper future-work only (one sentence, memo §8 wording verbatim).
**Start gate:** G0 artifact exists AND Track A merged. **Filter (c):** C1–C2
are cheap and de-risk the expensive steps; C5 caps the downside.

**Non-negotiable architecture (decided; do not revisit):**
- Home `src/lunaris/optimization/differentiable/` — NOT under
  `surrogate/st_lrps/` (consumes the runtime artifact through its public
  contract; keeps a future torch-SH baseline surrogate-agnostic; keeps the
  "ST-LRPS runtime (inference path) stays light" contract intact).
- **No retrofits:** `core.propagation` (scipy/Numba) and the batch torch
  propagator (inference-tuned, `.item()`/`detach` sites) stay untouched.
  Import only `surrogate.st_lrps.runtime.force_model` + `shared`/`common`.
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
torch. Gate test
`tests/test_diffprop_parity.py::test_zero_dv_point_mass_parity_vs_cpu`:
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

**C4 — single-impulse targeting + gradient validation (M).** One ΔV∈ℝ³ at
t₀; terminal-miss loss over ≤1 orbit; Adam and L-BFGS. Tests
(`tests/test_diffprop_gradients.py`, CPU-only where possible): central-FD vs
autograd per component (~1e-6 relative, float64, point-mass); strict loss
decrease for k steps; analytic anchor — residual disabled ⇒ optimized ΔV
matches the closed-form two-body answer. Horizon-vs-gradient-quality study is
a memo exhibit, not a unit test. All solutions reported as **local**.

**C5 — pre-registered kill criterion (write BEFORE running).** If autograd
does not beat an FD-based optimizer baseline (FD + Nelder-Mead/SLSQP) on
wall-clock **or** robustness for the MVP problem, write up the negative
result and stop the track. Gradient checkpointing / adjoint only after a
**measured** memory wall — never preemptively.

---

## 8. Deferred — gates only, no work orders

- **RL environment** (`lunaris/envs/`, `[rl]` extra): do not start before the
  paper is submitted. When the gate opens, hard requirements from memo §6
  apply: zero-action rollout bit-identical to plain `propagate()`; deadband
  station-keeping baseline mandatory before any RL claim; RIC observation
  reuses the single `analysis/ensemble/statistics.py` convention; gymnasium
  never enters core deps; framing is "research/demo simulation environment",
  station-keeping only. The detailed design lives in the source memo — do not
  re-derive it before the gate opens.
- **UQ Results-zone UI panel:** fold into the existing UI roadmap P1a
  (Results-zone figure gallery) — the UQ report already writes PNG figures +
  manifest into a run directory, which is exactly what P1a displays. No
  standalone task.
- **Run-comparison dashboard / artifact inspector UI:** wait for ≥2 accepted
  benchmark runs (schema stability), i.e. after G0 has run at least twice.
- **Full-P₀ (`covariance_6x6`) config wiring:** deferred until a concrete
  scenario needs a correlated initial covariance (see §3).

---

## 9. Execution queue (small-model order, cheapest de-risk first)

| # | Task | Needs GPU/artifact? | Rationale |
| --- | --- | --- | --- |
| 1 | B1 | no | tests-only, zero production risk, upgrades the methods sentence |
| 2 | A1 | no | the "was N enough?" paper figure; pure analysis over existing archives |
| 3 | A2 plumbing + self-comparison tests | no | de-risks the study for free |
| 4 | B2 (Linux/HPC) | no | external-friction track, run in parallel; 2-week go/no-go |
| 5 | **G0** (maintainer-scheduled) | yes | gates all paper integration |
| 6 | A2 real study; B3 | yes / no | consume the G0 artifact and B1–B2 results |
| 7 | C1 → C5 | C3+ yes | research bet, kill criterion armed |
| 8 | §8 items | — | only after their gates open |

**Definition of done for the whole plan:** the paper carries (F-UQ) one
ellipsoid/RIC figure + (F-conv) the A1 convergence figure + (T-prov) the
provenance table, backed by a G0-accepted benchmark;
`docs/FORCE_MODEL_VALIDATION.md` cites at least the B1 independent-path tier;
Track C has either a working prototype memo or an honest negative-result
memo. Everything else is explicitly deferred, not silently pending.
