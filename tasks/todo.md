# Periodic Evaluation During Training

Optional, non-invasive monitoring feature. Disabled by default. Must not touch
optimizer / scheduler / GradNorm / gradients / RNG / checkpoint selection / model.

## Plan

- [ ] 1. New module `st_lrps/training/periodic_eval.py`
  - `compute_periodic_eval_epochs(total_epochs, count, every_epochs, start_epoch=1)` (pure)
  - `build_periodic_eval_command(...)` (pure) -> argv list
  - `resolve_periodic_eval_plan(cfg)` -> plan (enabled, epochs, dataset, prefer, ...)
  - `load_periodic_eval_history(run_dir)` -> set of succeeded epochs
  - `run_periodic_eval(cfg, layout, epoch, plan)` -> bool (subprocess, logs, jsonl record)
- [ ] 2. `TrainConfig` fields + CLI args (config.py); mutual-exclusivity check
- [ ] 3. Evaluation CLI: add `--max-samples`, `--checkpoint-prefer {best,last}` (cli.py)
- [ ] 4. Engine hook in `train()` after ckpt_last save + history write (engine.py)
- [ ] 5. Studio UI collapsed "Periodic Evaluation" card + command preview (training_pages.py)
- [ ] 6. Tests (`tests/test_st_lrps_periodic_eval.py`) + run `pytest tests`

## Schedule semantics
- count=10, total=400 -> [40,80,...,400]
- every=25, total=100 -> [25,50,75,100]
- count/every mutually exclusive; both disabled -> empty schedule
- resume: skip epochs <= last completed and any epoch already recorded success

## Output convention
- `<run_dir>/periodic_evals/epoch_XXXX/` (eval outputs + eval_stdout.log/eval_stderr.log)
- `<run_dir>/periodic_evals/periodic_eval_history.jsonl`

## Subprocess invocation
`python -u -m st_lrps.evaluation.cli --model-dir <run_dir> --data <ds> --out <epoch_dir>
 --checkpoint-prefer last --max-samples N --batch-size N --device auto`
Failure does not abort training unless `--periodic-eval-continue-on-fail` is off.

## Review

Implemented and verified. Feature is optional and OFF by default; with no
periodic-eval flags the training command and behavior are byte-for-byte unchanged.

Validation:
- 23 new unit tests pass (schedule / plan / command builder / resume history /
  mocked runner success+failure+skip / UI flag emission).
- Full suite: 666 passed, 2 pre-existing skips, no new warnings.
- Real end-to-end smoke (tiny synthetic lunar h5, 5 epochs, --periodic-eval-count 2):
  ran at epochs 2 and 5 on ckpt_last, wrote summary_metrics.json + history jsonl,
  training completed normally.
- Real resume to epoch 7 (count 4): schedule correctly resolved to [7] only;
  past evals (2,5) not repeated; history clean (2,5,7 all success).

Files changed:
- st_lrps/training/periodic_eval.py (new)
- st_lrps/training/config.py (fields + CLI args + mutual-exclusivity guard)
- st_lrps/evaluation/cli.py (--max-samples, --checkpoint-prefer; evaluate(prefer=))
- st_lrps/training/engine.py (schedule resolve + post-ckpt_last hook)
- st_lrps/ui/studio_parts/training_pages.py (collapsed card + build_args + profile)
- tests/test_st_lrps_periodic_eval.py (new)

Limitations:
- Synchronous only (no async eval) — acceptable per spec.
- Single dataset per run (no comma-separated multi-dataset) — per spec.
- No Monitor "last periodic eval" status card yet; relies on log output (spec allows).

---

# Orbit-Level Benchmark in Studio

Goal: bring the gravity-model comparison harness into st_lrps and drive it from
a new Studio Analysis page (RK4 step, RK8/DOP853 selection, model selection, ...).

Decisions (user): hard move (no shim), expose BOTH modes (DOP853 + GPU RK4).

Done:
- git mv validation/gravity/compare_gravity_models.py -> st_lrps/evaluation/
  (header + docstring examples updated to st_lrps.evaluation.compare_gravity_models).
- Docs updated: root README.md, validation/README.md, validation/gravity/README.md.
- Tests updated: test_repo_hygiene.py + test_validation_docs.py to new module path.
- New Studio page st_lrps/ui/studio_parts/orbit_benchmark_pages.py
  (OrbitBenchmarkTab + OrbitBenchmarkPage): run-mode selector (DOP853 RK8 vs
  GPU batch RK4), model checkboxes + truth, RK4 step, DOP853 tolerances,
  scenario params, ST-LRPS dir, output dir, command preview, ProcessPane + gallery.
- Registered under Analysis (page index 5) in main_window.py; re-exported via studio.py.
- Tests: tests/test_st_lrps_orbit_benchmark_ui.py (7 tests) — relocation + both
  modes + nav registration.

Validation: full suite 673 passed, 2 pre-existing skips. Harness runs at new
path (`python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help`).

---

# Orbit-Level Benchmark — enhancements

User asks: (1) create/add custom comparison models in the UI; (2) selectable
ground-truth integrator RK45/DOP853; (3) CPU parallel processing; (4) GPU
fixed-step light/medium/robust method choice.

Harness (st_lrps/evaluation/compare_gravity_models.py):
- GPU fixed-step integrator dispatch: light=RK2 midpoint, medium=RK4 classic,
  robust=RK4+Richardson extrapolation (backend-agnostic gpu_fixed_step_advance);
  --gpu-integrator {light,medium,robust}. Verified orders 2/4/5.
- --truth-integrator {RK45,DOP853} via _cfg_with_integrator; applied to truth in
  both modes (GPU build_truth_trajectory_set + CPU run_random_scenario_mode +
  selected/worst plotting + batch-RK4 truth rebuild).
- --workers N: ProcessPoolExecutor with per-worker initializer rebuilding
  ephemeris + gravity caches; parallelizes the per-model adaptive CPU sweep
  (sequential fallback for workers<=1 / batch-RK4). Verified end-to-end (2 workers).

UI (orbit_benchmark_pages.py):
- "Add model" field+button -> custom shNN models (validated, persisted), dynamic grid.
- Truth integrator combo (always), Compare integrator (CPU), CPU workers spin,
  GPU method combo (light/medium/robust); mode-dependent enable; flags emitted
  per mode; persistence extended. _try_add_model is dialog-free for testability.

Tests: extended tests/test_st_lrps_orbit_benchmark_ui.py (15 total) — integrator
order/accuracy, fallback, CLI parse, _cfg_with_integrator, UI flags, custom-model
add/reject. Full suite: 681 passed, 2 skips. Tiny real CPU-parallel run verified.

---

# Accumulation + AIAA relocation + professional PDF report

Decisions: accumulate via resume toggle + total N (both modes); move AIAA code ->
evaluation and outputs -> outputs/ (gitignored); professional PDF template.

Done:
- GPU-mode accumulation/resume in run_gpu_batch_compare_mode: load stored
  per-scenario metrics, skip scenarios already done for all requested models,
  build truth + propagate only NEW scenarios, merge+write the union, aggregate
  over all. Plotting receives only newly-run scenarios (index-safe); aggregate
  tables/bars cover the cumulative set. Helpers _read_csv_rows/_coerce_numeric_row.
  CPU mode already accumulates via --resume (same seed, larger N).
- Studio "Accumulate previous results (resume)" checkbox -> emits --resume; persisted.
- AIAA: ported generate_publication_plots_v2 -> st_lrps/evaluation/publication_plots.py
  (argparse paths --run/--stlrps-run/--multi-run/--out-dir). Moved the whole
  "AIAA SciTech/" (15 MB, 79 tracked files: scripts + figures + run dirs) to
  outputs/aiaa_scitech/ (gitignored) and git rm'd from the index. README tree updated.
- Professional PDF report: _ReportPager toolkit (cover page, navy header band +
  accent rule, footer w/ page numbers + timestamp, styled tables w/ header shading
  + zebra rows + highlight, captioned figure pages, notes page). Rewrote
  write_report_pdf + write_gpu_batch_report_pdf; numeric content unchanged.

Validation:
- Real CPU-forced GPU-batch accumulation smoke: 2 -> resume -> 4 scenarios,
  aggregate n_ok=4, summary total=4/new=2.
- PDF reports render (valid PdfPages, ~74-84 KB).
- Full suite: 685 passed, 2 skipped (deterministic across 2 runs).

Note: the torch Laplacian double-backward tests in
test_surrogate_architecture_upgrades.py are natively fragile under full-suite
co-execution (MKL/OpenMP threading vs torch double-backward). The pandas-importing
publication test and the matplotlib PdfPages report test now run in isolated
subprocesses so they don't perturb that shared-process native state.
