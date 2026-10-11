# Changelog

## Unreleased

- **Fixed: Mission Studio tide values and absolute tolerance reach the run.**
  - k3 tides with a blank k3 field built a command the runner refused (`--tides-kind k3 requires an explicit --tide-k3 value`, exit 2); the engine has no default k3. The UI now refuses to build it and names the field.
  - A Love number of 0 (accepted by the CLI, `>= 0`) was dropped as invalid; it is now sent. Non-finite values are skipped with a warning; the reference radius must still be positive.
  - The Propagation page's absolute tolerance never reached single runs, batch runs or the pre-flight check (they read `solver_cfg.atol`); the page value now wins and `solver_cfg.atol` fills a blank field.
  - Ported from Sidera (66237c40).

- **Fixed: SciPy event root-finding crash on circular orbits.**
  - **The crash.** On an exactly circular orbit, RK45, Radau and LSODA raised `ValueError: f(a) and f(b) must have different signs` from inside `solve_ivp`. This was reproduced on a 100 km lunar orbit.
  - **Cause.** The apsis event `r . v` sits at rounding level there. SciPy detects a crossing from the solver states, but `brentq` re-evaluates the step interpolant, whose end values can have the other sign.
  - **The fix.** Events are now wrapped so a root search sees the step-end signs SciPy detected, and an event that starts on its surface stays silent until it leaves the rounding band.
  - **Results.** Runs without the inconsistency are bit-identical, tested on an eccentric orbit for RK45 and DOP853.
  - Ported from Sidera (7e3deea4).

- **Changed: albedo and thermal IR are integrated over the visible disk by default (default results change).**
  - **The defect.** The 18 x 36 point-facet sum (facets up to ~300 km across) gave 0.39x-1.25x of the converged acceleration at 100 km and 0.04x-1.05x at 30 km on a sunlit disk. The reference was a brute-force cap integral plus the exact uniform-exitance limit `C A/(m c) M (R/r)^2`.
  - **The fix.** `AlbedoConfig` / `ThermalConfig` gain `surface_integration` (default `"visible_disk_quadrature"`), `quadrature_radial_nodes` (16) and `quadrature_azimuth_nodes` (32). The new `lunaris.physics.surface_radiation_quadrature` is ported from the Sidera fork and re-verified independently. It integrates in the spacecraft's solid angle: exact for uniform exitance, and within 2e-4 of the full-Sun magnitude where the terminator crosses the disk. An azimuth recurrence replaces per-node trig, so the kernel costs about as much as the old sum.
  - **Scope.** The cached RHS kernel, the surrogate-gravity RHS, `get_acceleration_vector_breakdown` and the `albedo_accel` / `thermal_accel` wrappers all use it. Batch manifests record `albedo_surface_integration` / `thermal_ir_surface_integration`.
  - **Reproducing old results.** `surface_integration="facets"` restores the former path bit-for-bit (the 18 recorded RHS golden scenarios are unchanged).
  - **Impact.** 1 day at 100 km, polar: at most 5.7 mm for A/m = 0.002 m^2/kg and 57 mm for 0.02 m^2/kg, mostly in-track and from thermal IR.
  - **Registry.** Registered as `LUNARIS-MODEL-RAD-002/003` plus `LUNARIS-ALG-NUM-001` (Gauss-Legendre via NumPy).

- **Fixed: the engine's `simple` albedo kernel ignored `AlbedoConfig.enable_eclipse`.**
  - The cached RHS kernel, the surrogate RHS and the acceleration breakdown hard-coded the Moon-shadow test on.
  - They now follow `enable_eclipse`, as `surface_effects.albedo_accel` already did. The default (`True`) is unchanged.

- **Fixed: the strict solid-tide kernel depended on numba compile order.**
  - `solid_tides.accel_solid_tides_numba` and `solid_tide_accel_degree_numba` did not pin `fastmath`. When the fastmath wrapper `third_body_effects.accel_solid_tide` compiled first, the strict RHS inherited fastmath through the on-disk cache: 8913 of 12000 components changed, by up to 2.7e-15 relative.
  - Both kernels now pin `fastmath=False`. Output is bit-identical in every compile order.
  - A package-wide test fails if any unpinned non-inline helper compiles under a fastmath caller.

- **Fixed: the numba CUDA SRP shadow matches the CPU again.**
  - The device shadow factor still used the old smoothstep penumbra; the CPU now uses the exact visible-disk (lens-overlap) fraction. The GPU penumbra differed by up to 2.6% of full-sun SRP.
  - `_shadow_factor_conical_cuda` now mirrors the CPU kernel branch for branch. Under the CUDA simulator the shadow factor is bit-identical in full sun, penumbra, umbra and annular geometry for both the Moon and the Earth. SRP acceleration agrees to 2.8e-16 of full-sun SRP.
  - The CUDA SH longitudinal term uses the exact `1/rho^2` above the polar cutoff, as the CPU does. It is bit-identical for rho >= 0.13 mm and now agrees on and near the axis.
  - GPU SRP results change inside penumbrae only. Non-SRP GPU runs are bit-identical off the polar axis.

- **Fixed: worst-tail field statistics took one entry too many.**
  - The worst-1%/5% means sliced from `ceil(0.99*n) - 1`, which averaged 2 of 100 errors instead of 1.
  - They now take exactly `max(1, ceil(fraction*n))` entries.
  - The paper does not use these keys.

- **Fixed: frozen-search orbital elements are measured against the lunar equator.**
  - **The defect.** With a rotating Moon-fixed field (ST-LRPS screening, `--strict-frame`, `--paper-safe`, third-body validation), `elements_to_states` and the element histories used the J2000 integration frame. Inclination, node and argument of perilune, the `--i-deg` / `--argp-choices` sampling and every inclination/ω criterion were therefore measured from the Earth's equator, which is tilted about 23.5 deg from the lunar one.
  - **The fix.** Elements now live in the lunar equator of epoch, `R = q_i2f(t=0)` from the run's ephemeris. Stored states stay inertial. Identity-frame runs are bit-identical. The manifest records `element_frame`, its source and the matrix. Resuming a run written in another element frame is refused. Figures rotate the same way.
  - **Verification.** On DE440 the resulting lunar pole is RA 271.23 deg, Dec 67.98 deg at 2026-01-01, within 0.03 deg of the IAU 2015 model including its 18.6-year terms.
  - **Impact on existing runs.** Campaigns run before this fix (`v2_*`, `v3_*`, `v4_*`) report J2000 elements. For example, the v4 shortlist's J2000 inclinations of 42–136 deg are 52–64 deg and 113–124 deg against the lunar equator.
  - **Related fixes.**
    - ST-LRPS candidates are now always validated in the Moon-fixed frame.
    - The element <-> state GM comes from the validation gravity file header instead of a hard-coded constant.
    - `ElementBounds` rejects e >= 1.
    - A `strict_frozen`/`quasi_frozen` label now requires the validation history to cover the configured mission duration.

- **Fixed: the ST-LRPS CPU provider path honours `strict_domain` and no longer computes a discarded potential.** `SurrogateGravityModel.from_model_dir` now forwards `strict_domain` (and a new `paper_safe` flag) to the canonical runtime loader, so a strict provider raises on the CPU `acceleration_fixed*` path below the training envelope instead of only warning; non-strict loads still warn. A `paper_safe` load never falls back to the legacy local path. `acceleration_fixed` / `acceleration_fixed_batch` return the same acceleration bit for bit without the potential pass (planD_dirfix model, SH25 baseline, CPU: 3.2x per single call, 10x for N=2000). A `mu_override` more than 20% from the artifact's `resolved_mu_si` raises instead of being ignored. Empty `(0, 3)` batches pass the CPU and tensor domain guards. `import lunaris.surrogate.runtime` no longer imports torch. Tests are in `tests/test_surrogate_runtime_provider_fixes.py`.
- **Fixed: ST-LRPS `--resume-from` failed every time.** A new scaler recorded `dataset_content_sha256 = None` when the dataset contract did not embed the hash, while the reuse check computed it from the file. `training_config_hash`, which changes with `--epochs` and `--resume-from`, was also a critical config field. A checkpoint from before the output envelope also differed from its rebuilt config on `output_envelope_power`, because the field was absent on one side. The scaler now records the hash it is checked against, the config hash is no longer critical, and a missing `output_envelope_power` compares as `0.0` (envelope off). The end-to-end resume test now fails on a non-zero exit instead of skipping.
- **Fixed: ST-LRPS epoch losses were means of per-batch means.** The validation loader keeps its last, smaller batch, so that batch was over-weighted in `val_total_loss`, which picks the best checkpoint. All epoch statistics, including `loss_rel`, `rel_a_mean` and `wtz_frac`, are now sample-weighted. When the last validation batch is smaller than the others, reported losses and best-epoch selection can differ from earlier runs.
- **Fixed: ST-LRPS training-loop robustness.** The changes are:
  - A NaN/Inf validation loss writes `failure_manifest.json` and marks the run failed. Before, it raised from checkpoint scoring and left the run marked "running".
  - Resuming from an older snapshot drops the history rows and periodic-eval records of the epochs it re-trains.
  - Each run attaches and removes its own `train.log` handler. Before, `logging.basicConfig` wrote no log for a second run in the same process.
  - A training split smaller than one batch is refused. Before, it trained zero batches per epoch.
  - An independent `--val-data` file is validated like the train file.
  - Independent-file split manifests record `train_file_sha256` and `val_file_sha256`.
- **Fixed: ST-LRPS artifact checks.** `load_scaler_for_run` now checks a checkpoint's pinned `scaler_hash` when the checkpoint has no embedded scaler. `read_artifact_contract(strict=True)` now raises `ArtifactContractError` for an unreadable `config.json` or an unloadable or mismatched scaler, where it used to fall back silently. Existing runs load unchanged.
- **Fixed: GradNorm `update_interval=1` updated only on the first step.** It now updates every step, and `update_interval < 1` is rejected. Intervals of 2 or more behave as before.
- **Fixed: a malformed `w0_bands_tensor` checkpoint buffer was swallowed on load.** A buffer with the wrong type, shape or length, or with non-finite values, now raises `ValueError`. Checkpoints with a valid buffer, or with none, load as before.
- **Fixed: GPU terrain batches missed impacts in multi-step descents.** The torch and Numba CUDA terrain check bracketed each step against the terrain envelope sphere and skipped steps that started inside it, so a descent spanning several steps flew through the ground. A step that starts inside the envelope now enters at `alpha = 0`. Tests: `tests/test_terrain_multistep_descent.py` (torch CPU/CUDA, Numba CUDASIM, against a SciPy event reference).
- **Fixed: Numba CUDA t = 0 impact in terrain mode used the mean sphere.** The host mask now evaluates `|r| <= terrain_radius + impact_alt` with the device terrain sampler, as the torch backend does.
- **Fixed: terrain maps were sampled in MOON_PA axes.** LOLA topography is on mean-Earth/polar axes (DE421). `lunaris.common.lunar_map_frame` builds the DE440 PA->ME rotation (103.9 arcsec, about 875 m on the surface; matches SPICE `pxform` to 5e-20). The CPU impact event and telemetry, the torch samplers and the CUDA kernel apply it; the CUDA topo meta grows from 13 to 22 slots. Terrain impact times and points change for MOON_PA runs. Tests: `tests/test_lunar_map_frame.py`.
- **Fixed: Numba CUDA SRP shadow.** The cylinder test measured its projection from the Sun, so SRP was also off on the sunlit side behind the lunar disk: 2.0x the CPU eclipse time on a beta = 0 orbit. It had no Earth eclipse, no penumbra, and used `R_MOON` = 1738 km. The kernel now mirrors the CPU conical shadow factor with the `SRPConfig` radii, eclipse switches, AU and P0. GPU SRP results change. Diagnostics report the CPU shadow labels.
- **Fixed: Numba CUDA SH near the polar axis.** The kernel used the CPU's 1e-12 m longitude cutoff instead of 1e-6 m. Inside the `rho^2 < 1e-24` cutoff it now uses the analytic m = 1 transverse limit, as the CPU does. Results off the axis are bit-identical.
- **Fixed: Numba CUDA quaternion SLERP extrapolated past the attitude table end.** It now holds the end rows like the CPU interpolator.
- **Fixed: chunked SciPy runs restart from the chunk end.** A chunk whose end was off the output grid restarted from the last output sample, so the span was integrated twice and events in it were logged twice; a chunk without events also crashed the event concatenation. Chunk ends are now always sampled, only grid samples are kept and boundary events are de-duplicated. A stop file present before the start now stops an unchunked SciPy run at t0, and the between-chunk stop-file poll no longer depends on `stop_event_in_scipy`. Tests: `tests/test_scipy_chunked_restart.py`.
- **Fixed: no fake apoapsis at t ~ 1 s.** The periapsis/apoapsis events returned +1 while the absolute time was below 1 s, so a run starting with r·v < 0 logged an apoapsis at ~1 s and a real apsis in the first second was hidden (and the guard did nothing for t0_s > 1 s). The guard is now start-relative (NaN, default 0 s) and a start on an apsis is not an event. Event times match Kepler for DOP853 and RK4. Tests: `tests/test_apsis_event_start_rule.py`.
- **Fixed: a longitude-crossing event fires once per revolution.** `wrap_pi(lon - lon0)` jumps at the antipode, which solvers logged as a second crossing; the root function is now `sin(lon - lon0)` on the near half and NaN elsewhere, with a start-relative NaN guard (default 0 s). Tests: `tests/test_longitude_crossing_once_per_rev.py`.
- **Fixed: SPICE tables cover `[0, t0_s + duration_s]`.** They covered `duration_s`, so with `t0_s > 0` the Sun/Earth vectors and the attitude froze for the last `t0_s` seconds; a negative `t0_s` is refused. Table builds now furnish and unload only their own kernels (`spice.unload`) instead of `kclear`-ing the whole pool by default (`SpiceBuildConfig.clear_kernels_after` now defaults to False), hold a process-wide SPICE lock, and normalise a timezone-aware `start_date` to UTC. `propagate` refuses a window outside the ephemeris table and a start below the impact sphere when impact detection is on. Tests: `tests/test_ephemeris_build_window.py`, `tests/test_propagator_start_guards.py`.
- **Fixed: batch runs refuse `time.t0_s != 0`.** The GPU/torch loops start at t = 0 and the CPU path resampled a `t0_s`-offset solution onto the 0-based archive axis. summary_only mode no longer runs one sample per sub-batch (its host budget was 0); the summary is identical. Tests: `tests/test_batch_engine_time_origin_and_summary_cap.py`.
- **Fixed: batch archive writers.** `output_path_resolved` now carries the format suffix (an npz run at `*.h5` broke the `.part` rename), part files carry the PID so concurrent runs on one path cannot delete each other's work, and `BatchPropagationConfig.overwrite_output` (default True, replacement now logged) refuses an existing archive when False. HDF5 epoch-block reads decompress each chunk once (bit-identical, 6-9x faster in a 1000x5000 benchmark). Tests: `tests/test_batch_archive_writer_paths.py`, `tests/test_hdf5_epoch_block_reads.py`.
- **Fixed: adaptive-degree `min_degree=0` is honoured** (it was replaced by 4) and a non-positive `power` raises instead of becoming 2.5. Tests: `tests/test_adaptive_degree_explicit_defaults.py`.
- **Fixed: a NaN void in a float LDEM no longer collapses the terrain envelope to the mean sphere.** Tests: `tests/test_topo_envelope_nan_voids.py`.
- **Fixed: Slurm templates.** They sourced `env_template.sh` next to the spooled script copy and died at the first line; they now use `LUNARIS_HPC_DIR` (exported by `hpc/submit.sh`) or `$SLURM_SUBMIT_DIR/hpc`. The training templates `exec` their program and the scenario launcher relays SIGTERM/SIGUSR1/SIGINT to the trainer, so `--signal=B:TERM@600` reaches it. The batch array template refuses unscrambled `sobol`. Tests: `tests/test_hpc_signal_relay_and_templates.py`.
- **Performance: the CPU lunar RHS compiles once per process and is cached on disk.**
  The numba RHS kernels were closures over the run's arrays and settings, which
  numba freezes as constants, so every `DynamicsEngine` (each CPU batch sample,
  each validation run) recompiled them: 1.6-2.5 s per engine for GL1800F
  degree 50/200, SH-only and SH + Sun/Earth. They are now module-level
  `cache=True` kernels that read one `_RhsPack` (a numba StructRef). Results are
  bit-identical (`np.array_equal`) on 27 force configurations: 1000 RHS states
  and a 1-day DOP853 trajectory each. The surrogate (Python) RHS is unchanged.
  Numba checks a cached kernel only against `engine.py`; a stamp next to the
  cache hashes the 11 callee modules (and the constants numba freezes into
  them), and on import a mismatch removes engine.py's cache files so the
  kernels recompile against the current force code.
- **Fixed: YAML benchmark configs are parsed by PyYAML only.** A hand-written subset parser stood in when PyYAML was missing, so the same file parsed differently per machine. Under the subset parser, `rtol: 1e-10` was a float (a string under PyYAML), `n: 5  # five` kept its comment, and `seed: 010` was 10 (8 under PyYAML). Without PyYAML, a YAML config now raises `BenchmarkConfigError`. JSON configs still need nothing extra.
- **Fixed: honest ST-LRPS and third-body tests.**
  - `tests/test_st_lrps.py` had 35 `except ImportError: print("[SKIP]"); return` guards, which let a broken in-repo import pass as green. The in-repo imports are now unguarded. Optional `h5py` uses `pytest.importorskip`.
  - Four `_test_*` functions that pytest never collected (scaler round-trip, chain rule, direction mask, artifact resolver) are renamed to `test_*`, and all four pass.
  - `tests/test_third_body_effects.py` dropped its local constant fallback, whose `MU_SUN` (1.32712440018e20) disagreed with the SSOT (1.32712440042e20).
- **Fixed: the perturbation-budget summary labels its geometry honestly.** Sampling always uses synthetic Sun/Earth geometry, but the summary printed `ephemeris_requested` whenever the synthetic *gravity* fallback was off. The summary now always reads `synthetic_geometry`. `use_ephemeris=True`, which was silently ignored, now raises `NotImplementedError`. The config field that gates synthetic gravity coefficients is renamed from `use_synthetic_geometry_fallback` to `use_synthetic_gravity_fallback`. The CLI flag `--synthetic-fallback` is unchanged.
- **Fixed: report history extractors read the state layout from the time grid and stop guessing units.** Two bugs are fixed in `extract_rv_vectors` / `extract_elements`:
  - A row-major `(N, 6)` state with `N >= 6` was read as `(6, N)`. Now, when both axes could hold the state, the time grid decides. The canonical `(6, N)` layout wins for a square state, and a shape that matches neither axis raises.
  - Magnitude-based unit guesses (`_maybe_km`, `_maybe_r_m`, `_maybe_v_mps`) are removed. Histories are SI by contract, and canonical `*_km` / `*_kmps` keys are kept as stored. Before, an `r_norm_km` above 1e5 km was divided by 1000 again.
- **Fixed: the step-size convergence gate (`lunaris.validation.convergence`).** Three changes:
  - It is now one-sided (`observed >= expected - order_tol`), as its docstring always said. Superconvergence no longer fails.
  - The reference-based order comes from the finest (medium, fine) error pair instead of the coarse pair, which could hide a stalled sequence.
  - 0/0, a solution that does not move across the sweep, now gives order `nan` and fails instead of passing as "converged".

  Checked against RK4 (order 4.0) and Euler (order 1.0) on `y' = y`.
- **Fixed: `lunaris-batch` exits 2 when the run produced no usable result.** If every sample failed or the metrics could not be built, the runner used to print `[BATCH][WARN]` and exit 0. It now prints `[BATCH][FATAL]` and exits 2.
- **Fixed: the `lunaris` single run no longer reports success when the integration fails.** A failed SciPy integration returns a partial trajectory instead of raising. The runner used to print `[OK] Finished.` and exit 0 for it. It now keeps `run_config.json` / `run_diagnostics.json`, skips the report, prints `[FAIL] Integration failed at t = ...`, and exits 1. A requested SH degree above the gravity file's maximum was silently evaluated at the file maximum while the requested degree was recorded. The runner now warns and records the effective degree in the run config, report meta and telemetry, plus `requested_sh_degree` / `effective_sh_degree` in the run diagnostics.
- **Fixed: `--samples-per-period` now sets the output grid.** The default config's `output_dt_s = 60 s` used to win, so the flag was recorded and ignored. On `lunaris` (single run), the flag now clears `output_dt_s` and the time-grid plan derives the spacing from the osculating period. `lunaris-batch` applies the same rule in the runner, because the batch engine would otherwise fall back to `10 * batch_dt_s`. The flag can no longer be combined with `--output-dt-s` on either command.
- **Fixed: `compute_ensemble_statistics(use_survived_only=True)` no longer overwrites `result.valid_mask`.** It used `&=` on the array that `valid_sample_mask()` returns, which is the stored mask itself. Impacted samples then stayed marked invalid for every later analysis of the same result.
- **Fixed: gravity-reference hash pins on Windows checkouts.** `.gitattributes` now pins `validation/gravity_reference/generators/**/*.py` to LF, so the rotating-validation runner hashes match the evidence matrix on a CRLF checkout. The `synthetic_degree4` and `grail_degree32` manifests carried stale `generator.script_sha256` values. Rerunning both generators reproduced their data and coefficient files byte-for-byte, so their regenerated manifests are committed. The only change is the script hash. `tests/test_gravity_reference_generator_hashes.py` gates every manifest.
- **Fixed: ST-LRPS evaluation leakage scope.** `evaluate()` now excludes the training rows automatically when the evaluation file's `content_sha256` matches the run's in-file training dataset, and fails closed when `split_indices.npz` is missing. `--max-samples` counts retained rows. Reports and `summary_metrics.csv` carry `eval_scope` (`same_file_train_rows_excluded`, `heldout_file`, `training_file` or `unverified`) and `excluded_train_rows`.
- **Fixed: streaming evaluation statistics.** The streaming L-inf keeps NaN visible instead of returning 0.0. The altitude bins span the dataset's recorded altitude range, and under/overflow counts are reported. The streaming ratio-of-sums relative error is now `rel_ratio_of_sums_pct`; `rel_mean_pct` (in-memory per-point mean, read by `mean_relative_a_error`) keeps its meaning.
- **Fixed: benchmark trajectory interpolation.** `interpolate_state_to_times` uses cubic Hermite position and velocity (O(h^4)/O(h^3) instead of linear O(h^2)) and refuses to extrapolate. A model trajectory that does not cover the truth grid becomes a `failed_time_coverage` / `time_coverage` row instead of a held end state.
- **Fixed: field-validation and paper-evidence hygiene.** Field metrics are NaN when any prediction is non-finite. Split kinds are downgraded to interpolation unless the model was trained with the same split policy. Independent-file runs are hash-checked against the training file. The paper-evidence gate intersects the persisted split indices. The multi-seed `std` is the sample stdev (`None` for one seed). An unreadable evidence manifest is refused instead of being reset, and the manifest is written atomically. String booleans such as `"false"` are parsed in the training argv.
- **Fixed: dataset generator seeding and validation.** Chunk seeds come from `numpy.random.SeedSequence` (`chunk_rng_version=seed_sequence_v2`). Previously chunk *i* of seed *s* was chunk *i-1* of seed *s+1*. Single-cloud datasets therefore differ bit for bit from earlier ones; suite datasets are unchanged. All writers write `<name>.partial` and rename it when done. `validate_dataset_file` recomputes `content_sha256` and derives the label recompute tolerance from the storage dtype.
- **Fixed: benchmark output honesty.** Missing scenario statistics are left blank instead of being copied from other statistics. Cold, warm and JIT timings that were never measured are reported as `""` with `timing_protocol=not_measured`, and the validator accepts that. Cached runtime rows count only scenarios with a measured runtime. The phase-explained fraction compares variances over the same epochs. Provenance no longer records the first `*.pt` it finds as the model checkpoint.
- **Fixed: post-processing GM, radius and eclipse timing.**
  - GM and radius: `process_simulation_results` and `build_orbit_analysis`
    read `mu_m3s2` / `R_body_m` attributes that the dynamics engine does not
    have. They therefore always used a hard-coded 4.902801e12 m^3/s^2 and
    1737.4 km, or `MU_MOON`. They now use the run's gravity model
    (`engine.grav.GM_m3s2` / `R_ref_m`), the same source the propagator uses,
    and fall back to `MU_MOON` / `R_MOON`. Altitudes and elements in reports
    of SH runs change accordingly (GL1800F: altitude reference 1738.0 km
    instead of 1737.4 km).
  - Eclipse mask: it now interpolates the Sun/Earth ephemeris at each sample
    time. Before, it used the next table row. Its Moon radius is the SRP
    occulter radius when the engine has one.
  - The ephemeris index helper now returns the nearest-left row, as its
    docstring states. The ground track uses this helper.

  Tests are in `tests/test_analysis_gm_radius.py`.
- **Fixed: rv-to-COE edge cases.**
  - Inclination: `acos(hz/|h|)` lost precision near 0 and 180 deg (1e-6 rad
    came back 4e-5 relative off; below about 1.5e-8 rad it came back as 0).
    `atan2(|h_xy|, hz)` is now used for |cos i| >= 0.5.
  - Retrograde equatorial orbits: the longitude of periapsis and the true
    longitude are now measured in the direction of motion, so rv -> COE -> rv
    round-trips. Before, the round trip mirrored the orbit in y.
  - `wrap_angle_2pi` no longer returns exactly 2*pi for tiny negative angles.

  On 20,000 lunar orbits (i 40-140 deg, e 0.005-0.1), only the inclination
  output of `rv_to_coe_select` and `batch_y_to_elements` changes. It changes
  in 2,120 samples, all outside 60-120 deg, by at most 4.4e-16 rad (3 ulp).
  The frozen-search `screening_orbital_elements_vec` is a separate code path
  and is bit-identical. Tests are in `tests/test_math_utils.py`.
- **Fixed: albedo and thermal-IR wrapper inputs.** The simple backend of
  `albedo_accel` now honours `AlbedoConfig.enable_eclipse` like the facet
  backend and reads the canonical `albedo_const`. `AlbedoConfig` keeps its
  legacy alias `A_moon` equal to `albedo_const`, also after
  `dataclasses.replace(cfg, albedo_const=...)`. Before, the alias kept the
  previous value. In `temperature_grid` mode, `calc_thermal_ir_accel` (and
  `thermal_accel`) now raises `ValueError` when the per-facet temperature grid
  is missing or has the wrong length. Before, it returned a silent zero.
  Default results are unchanged. Tests are in `tests/test_surface_effects.py`
  and `tests/test_thermal_ir.py`.
- **Fixed: strict SHADR loading.** In strict mode every numeric field must now
  be a complete literal. Before, a regex fallback read `1.0-08` as the two
  numbers 1.0 and -8.0 and accepted `-9.0E-05junk`. Fortran `D` exponents
  parse as one number. A non-integral degree/order index or normalization
  state (`1.7`, `1.5`, `nan`) is rejected instead of truncated or skipped, in
  both `load_shadr_ascii` and `read_gravity_model_header`. ICGEM `.gfc` files
  are refused with a clear error instead of reaching the binary SHBDR parser,
  and `--gravity-file-path` no longer offers `.gfc`. The GL1800F table loads
  bit-identically (SHA-256 of the coefficient arrays at degree 1800, 660 and
  50 unchanged). Tests are in `tests/test_io_gravity.py`.
- **Fixed: SH acceleration within nanometres of the polar axis.** The numba and
  torch kernels softened the longitudinal projection to `1/(rho^2 + 1e-24)`,
  which halved it at `rho = 1e-12 m` (up to 49% transverse error against a
  finite difference of the potential); they now use the exact `1/rho^2` above
  the axis cutoff, and the longitude fallback uses the same `rho^2 >= 1e-24`
  test as the cutoff. The batch potential kernel dropped the whole transverse
  field for `1e-14 m < rho <= 1e-14 r` and now applies the analytic axis limit
  there. Off-axis results (`rho >= 0.13 mm`) are bit-identical (measured on
  205 points for the serial, parallel, batch and torch paths). Tests are in
  `tests/test_sh_pole_axis.py`.
- **Fixed: strict SH helpers no longer inherit `fastmath`.** Numba compiles a
  helper with the flags of whichever caller compiles it first and caches that
  build on disk. The helpers shared by the strict serial SH kernel and the
  fastmath kernels (`_compute_sh_acceleration_parallel`,
  `sh_potential_accel_batch_serial`), including `_kahan_sum_step` and
  `common.math_utils.clamp`, now set `fastmath=False` explicitly. Measured on a
  degree-200 field: before, the serial acceleration moved by up to 11 ulp
  (1.1e-15 relative) when a fastmath kernel compiled first, and the drift
  persisted through the cache; after, it is bit-identical in every compile
  order. Results compiled in the strict order are unchanged. Tests are in
  `tests/test_sh_fastmath_isolation.py`.
- **Fixed: SRP eclipse geometry.** The conical shadow factor is now the
  visible solar-disk fraction (lens area of the apparent solar and occulter
  disks, Montenbruck & Gill 2000, sec. 3.4) instead of a smoothstep across the
  penumbra cones. Against an exact spherical-cap integral it is within 3e-4 in
  `nu` (the smoothstep was up to 0.046 off near the Moon and returned ~0 in the
  annular region beyond the umbra apex). Full-sun and umbra limits stay exactly
  1 and 0. `SRPConfig.enable_earth_eclipse` now defaults to `True`, so SRP is
  dimmed during lunar eclipses like albedo and thermal IR; a zero Earth vector
  (no ephemeris) is ignored. Default SRP, albedo-eclipse and thermal-IR-eclipse
  results change inside penumbrae and Earth shadows. Tests are in
  `tests/test_srp_eclipse.py`.
- **Added: ST-LRPS direction-accuracy training options.** The changes are:
  - A relative acceleration loss.
  - Radial/cross-track splits of the acceleration error, in the same scaled units as the acceleration loss.
  - A worse-than-zero validation metric.
  - An optional radial output envelope `rho^(N_base+1)`, recorded in the architecture fingerprint.
  - Checkpoint-selection plumbing for the new terms.

  Tests are in `tests/test_st_lrps_direction_fixes.py`.
- **Fixed: batch archive metadata merge.** The ST-LRPS artifact block and the backend plan share keys such as `runtime_model_kind`. Splatting both into `write_metadata(**...)` raised `TypeError` after a completed surrogate propagation. `merge_archive_metadata` now merges the blocks, the later block wins, and differing duplicates are logged. Tests are in `tests/test_batch_archive_metadata_merge.py`.
- **Added: `formal/`.** A Lean 4 + Mathlib project that states and proves ST-LRPS design properties. The local Mathlib build cache is git-ignored.

- **Fixed: the frozen-orbit inclination-rate limits now scale with the arc
  length.** `FrozenClassificationConfig.for_mission_duration` derived every
  rate limit as envelope / duration except the inclination rate, which stayed
  fixed at `1e-10` / `1e-9` rad/s (strict / quasi). For a 60-day arc the fixed
  quasi limit was 1.8 deg/yr, against 12.2 deg/yr from the 2 deg envelope.
  The factory now also accepts `strict_inclination_range_max_rad` /
  `quasi_inclination_range_max_rad`. The screening score does not use this
  limit, so shortlists and scores are unchanged and only labels move. A
  full-force 4096-sample rerun reproduced an earlier campaign's shortlist,
  scores and validation metrics exactly and labelled 7 of 16 orbits
  quasi-frozen (previously 0).

- **Added: resumable batch propagation (`lunaris.core.batch_checkpoint`) and
  stage-1 resumability for `lunaris-frozen-search`.** A long ensemble
  screening run is a single `run_batched_fixed_step` call; when the process
  died (a crash, a reboot, or an unrelated `taskkill /IM python.exe`) every
  hour of it was lost, because nothing was written until the call returned.
  The shared batched fixed-step loop now accepts an optional
  `BatchCheckpointSpec` and persists, per device chunk, its integrator state
  at snapshot boundaries plus the snapshots produced since the previous save;
  a resumed call restores the tensors in their exact device dtype and
  continues the same RK4 arithmetic, so trajectories, impact bookkeeping and
  the output grid are identical to an uninterrupted run (tested bitwise,
  chunked and unchunked, with impacts). Checkpoints carry a contract (sample
  count, grid, effective dt, dtype, a hash of the initial states, impact
  settings); a mismatched directory is moved aside rather than reused.
  `lunaris-frozen-search` wires this into stage 1 and additionally persists
  every finished screening block, so `--resume` replays finished blocks and
  continues the interrupted one. New flag
  `--screening-checkpoint-minutes` (default 20, `0` disables); the work
  directory `stage1_checkpoint/` is deleted once `stage1_screening.npz` is
  written, and `manifest.json` records how many blocks were restored.

- **Changed: release/version hygiene.** 0.1.0rc2 is now tagged (`v0.1.0rc2`)
  and main carries `0.1.0rc3.dev0`, so a wheel built from main no longer
  advertises itself as the released rc2 while containing post-rc2 changes.
  `CITATION.cff` keeps naming the newest *released* version.
  `tests/test_release_metadata.py` now enforces this: a non-empty Unreleased
  section requires a `.dev` version, dev versions must not duplicate an
  already-released heading, and the citation version/date must match the
  newest changelog release.
- **Fixed (HIGH): frame mismatch in the force-sample benchmark
  (`--force-sample-trajectory`).** The ST-LRPS path fed inertial truth
  positions straight into the Moon-fixed surrogate field ("inertial ≈
  body-fixed approximation") and compared the resulting fixed-frame
  accelerations against the truth RHS's inertial accelerations — invalid for
  a non-axisymmetric lunar field once the Moon rotates over the sampled arc.
  Each sample now goes through r_I → r_F → a_F → a_I with the same ephemeris
  rotation the truth dynamics engine uses, so both sides of the comparison
  are inertial. Acceleration-error numbers produced by this mode before this
  fix should be discarded. (The GPU batch orbit benchmark was already
  frame-correct and is unaffected.)
- **Changed: fit-region benchmark honesty hardening (external review).** The
  adaptive surrogate series is now reported as `ST_LRPS_ADAPTIVE_<integrator>`
  (was `ST_LRPS_CPU_<integrator>`): the integrator loop is CPU SciPy, but the
  surrogate field evaluates wherever the loaded model lives, and the runtime
  rows now carry that split provenance (`device`/`dtype` = CPU float64
  integrator state; new `model_device`/`model_dtype` columns = the model's
  actual device/dtype). Config, CLI help, and docs no longer claim the series
  "isolates surrogate field error from integrator error" — it substantially
  reduces the fixed-step contribution but runs at looser tolerances than the
  truth, so the residual is not an independent decomposition
  (`docs/REPRODUCIBLE_BENCHMARKS.md` documents the missing control series).
  A run where only some adaptive scenarios finish now reports
  `status="partial"` (surfaced under `partial_models` in the summary) instead
  of a blanket `status="ok"`.
- **Added: `surrogate.require_st_lrps` benchmark-config key** (maps to
  `--require-st-lrps`). `configs/benchmarks/st_lrps_fit_region_sweep.json`
  sets it: the fit-region sweep now fails closed when no valid ST-LRPS model
  directory resolves, instead of silently degrading to an SH-only ladder.
- **Changed: `TelemetrySample.sample_kind` no longer defaults to
  `output_state`.** Every producer must declare the kind explicitly so a new
  producer cannot silently promote solver probes to trajectory science; only
  the decoder still maps a missing kind to `legacy_unknown`. The
  `telemetry.ndjson` artifact docstring now states that `rhs_probe` samples
  are stdout-only (they were never written to the artifact).
- **Changed: `tools/api_inventory.py` boundary scan fails closed on a file it
  cannot parse** (previously a `SyntaxError` silently skipped the file).
- **Docs consistency fixes:** `ALGORITHM_TRACEABILITY_POLICY.md` now matches
  CI (the registry audit runs as an advisory step, not "not run in CI");
  `PUBLIC_API.md` scopes the manifest claim to modules with a literal
  `__all__`; the 0.1.0rc2 API-boundary entry now states that only the
  `user-stable` tier is a stable surface at 0.x, matching `VERSIONING.md`.

## 0.1.0rc2 — 2026-07-17

- **Changed: API-boundary cleanup — internal helpers consumed across
  subsystem boundaries got public names.** (Only the `user-stable` tier in
  `docs/PUBLIC_API.md` is a stable surface at 0.x — see `docs/VERSIONING.md`;
  the modules renamed here are outside it, so no deprecation aliases beyond
  those noted.) Renames, old → new:
  `lunaris.common.math_utils._sample_grid_bilinear_kernel` →
  `sample_grid_bilinear_kernel`; `._sample_2d_scaled_bilinear_kernel` →
  `sample_2d_scaled_bilinear_kernel`;
  `lunaris.loaders.io_surface._grid_topo_payload` / `._grid_dem_provenance` →
  `grid_topo_payload` / `grid_dem_provenance`;
  `lunaris.physics.relativity_effects._schwarzschild_components`,
  `._external_schwarzschild_diff_components`, `._de_sitter_components`,
  `._external_1pn_components` → same names without the underscore;
  `lunaris.analysis.postprocess._extract_rv_vectors`,
  `._groundtrack_if_available`, `._first_present` → public names;
  `lunaris.core.propagation.time_grid._get_ref_radius_and_mu` /
  `._get_sh_degree` → `get_ref_radius_and_mu` / `get_sh_degree`;
  `lunaris.core.dynamics.preparation._provider_get` / `._provider_has` →
  `provider_get` / `provider_has`;
  `lunaris.surrogate.st_lrps.data.datasets._discover_dataset_name`,
  `._find_latest_dataset`, `._resolve_loader_worker_count`,
  `._resolve_lunar_dataset_contract`, `._build_train_val_indices` → public
  names (the `training.cli` fold keeps the old keys working for one MINOR);
  `lunaris.surrogate.st_lrps.data.splits._hash_indices` → `hash_indices`
  (compat alias kept for one MINOR);
  `lunaris.surrogate.st_lrps.networks.models._compute_harmonic_w0_bands` /
  `._get_output_head_params` → public names. Moves:
  `lunaris.batch.summary._osculating_elements` →
  `lunaris.common.math_utils.screening_orbital_elements_vec`; the spinbox wheel
  guard `_NoWheelOnSpinFilter` →
  `lunaris.ui_foundation.event_filters.NoWheelOnSpinFilter`.
- **Added: machine-checkable API boundary.** `docs/PUBLIC_API.md` gains a
  "Naming And Boundary Policy" section and now documents the full exported
  surface of `lunaris.core.propagation` and `lunaris.surrogate.runtime`
  (previously exported via `__all__` but undocumented). New gates:
  `tools/api_inventory.py` + `docs/api_snapshot.json` snapshot
  (`tests/test_api_snapshot.py`), PUBLIC_API-table ↔ `__all__` sync
  (`tests/test_public_api_doc_sync.py`), a ban on cross-unit imports of
  `_`-prefixed symbols in `src/lunaris` (`tests/test_api_boundaries.py`),
  and the import-linter contract `mission UI does not import ST-LRPS studio
  widget internals`.
- **Corrected Mission Monitor trajectory semantics.** Every structured sample
  now declares `output_state`, `accepted_state`, or `rhs_probe`. Adaptive and
  fixed-step replay artifacts are emitted from the exact solver-returned output
  states; transient adaptive stages/trials remain explicitly labelled live
  probes and are excluded from `telemetry.ndjson` and scientific trajectory
  widgets. Terminal event states are retained once, including non-grid event
  times, and chunk boundaries keep one monotonic sequence. Old v1/bare-JSON
  samples remain readable as `legacy_unknown` but are not presented as proven
  accepted states. Bounded build/serialization/writer/sink/terrain failure
  diagnostics are reported without allowing telemetry to stop propagation.
- **Added a Windows Python 3.11 core smoke gate.** The job builds and installs a
  wheel, checks `import lunaris`, the headless CLI help and `lunaris-data list`,
  optional-dependency boundaries, and external data-root path handling. It does
  not claim Windows CUDA or full PySide6/OpenGL validation.

- **Added the Mission Monitor (phases 1–3): a live/replay observation console.**
  New navigation page with a dockable widget workspace (Altitude/Radius,
  Orbital Elements, Integrator Health, Event Timeline, Backend & Provenance),
  workspace presets, and a shared-cursor replay timeline (play/pause, step,
  event jump, 0.25×–20×). Backed by a typed, versioned telemetry protocol
  (`lunaris_telemetry_v1`, `lunaris.common.telemetry_contract`): the propagator
  now emits `[TELEMETRY] {json}` samples (SI state, osculating elements with
  singularity-honest omission, sequence ids) and the CLI emits one
  `[TELEMETRY_META]` provenance line (config hash, git commit, gravity
  model/degree, μ). `--telemetry-artifact on` writes solver-returned output
  states to `telemetry.ndjson` in the run directory for replay (UI runs enable it).
  UI memory is bounded (ring-buffer store + min/max-envelope display
  downsampling); paint cadence is decoupled from telemetry cadence; legacy
  bare-JSON telemetry keeps working through a compatibility adapter. New
  import-linter contract keeps `lunaris.ui.monitor` off the physics hot path.
  Docs: `docs/MISSION_MONITOR.md`.
- **Completed the Mission Monitor (phases 4–6).** 3D Orbit View (reuses the
  Orbit Setup GL scene: Moon, display-decimated trace, cursor-following
  position marker, impact markers, explicit no-GL fallback) and State Vector
  (inertial/body-fixed with honest channel gating). Multi-tab dashboards with
  versioned layout persistence (`lunaris_monitor_layout_v1`,
  `monitor_layout.json`, atomic writes; corrupt files quarantine to `.bak`
  and can never block startup). Batch Progress widget consumes the existing
  `[BATCH_PROGRESS]`/`[BATCH_METRICS]` protocol (impact stats with the
  runner's 95% CI, requested→actual backend, ST-LRPS runtime kind) via
  controller-scoped payloads that never touch the live run's store.

Merged to `main` after 0.1.0rc1 (#115–#119): 100 files, ~12.6k insertions.
This is not a formatting-only delta — it adds a new `lunaris.validation`
convergence module, a full algorithm-traceability system, and widens the lint
gate. **`main` is materially ahead of rc1**; a release candidate cut from it
should be 0.1.0rc2.

- **Added an algorithm traceability system (#117, #118).** Every non-trivial
  algorithm in the codebase is registered in `docs/algorithms/algorithm_registry.yaml`
  (44 entries) with a persistent ID and a citation into `references/references.bib`
  (31 entries, each verified against a primary source locator). `tools/algorithm_registry.py`
  validates the registry against `schemas/algorithm_registry.schema.json`, renders
  `docs/ALGORITHM_CATALOG.md` deterministically (with a `--check` staleness gate),
  and audits the source tree for algorithm-ish code that is not registered. The
  authoring rules live in `docs/ALGORITHM_TRACEABILITY_POLICY.md`. The audit runs
  in CI as **advisory** (exit 0); `--strict` exists but currently reports 27
  uncovered files, so gating on it is blocked on triaging that list.
- **Added a convergence gate for the classical core (#117, #118).** New
  `src/lunaris/validation/convergence.py` plus `tests/test_convergence_gate.py`,
  `tests/test_sh_high_degree_stability.py`, `tests/test_altitude_binning.py`, and
  `tests/test_adaptive_blend_strict_guard.py` (adaptive-degree blending now fails
  closed rather than silently approximating). Supporting analysis tooling:
  `tools/blend_error_study.py`, `tools/gpu_qualification.py`.
- **UI: Results zone, shared shortcut SSOT, and CVD-screened plot series (#115,
  #116).** New `lunaris.ui.core.{results_index,shortcuts,plot_style}` — a run
  index behind the Results & Export page, one source of truth for keyboard
  shortcuts, and a multi-series plot style that pairs colour with a line-dash
  cycle so series identity survives colour-vision deficiency. Verification
  tooling: `tools/ui/color_audit.py`, `tools/ui/spacing_scan.py`,
  `tools/ui/snapshot_suite.py`.
- **Widened and documented the lint gate (#119).** `ruff` `select` grew beyond
  `["E","F","W","I","UP","B"]`, and — more usefully — the rules that are
  deliberately *not* selected now carry an in-file rationale so the decision is
  not re-litigated every round. Note the pin: CI runs `ruff==0.12.0`, so
  validating against a newer local ruff is misleading.

## 0.1.0rc1 — 2026-07-11

First release candidate after the month-long feature freeze and validation
campaign. Highlights: canonical batch/ensemble terminology and engine refactor;
physics-audit guard closures (RKN4 × velocity-dependent forces, post-terminal
event recording, symplectic × adaptive-degree gravity); a trust-gated
deserialization boundary for `.pt` artifacts (`safe_torch_load`,
`--trust-artifact`, `LUNARIS_TRUST_ARTIFACT`); dataset-hash provenance
normalization (`content_sha256` canonical); ST-LRPS capability SSOT, full-field
SH baseline, compute accounting, and training-loop hardening (gradient
accumulation scaling, FP32 validation, collocation cadence); the `lunaris-data`
manifest-driven asset CLI; UI design-system consolidation. The ST-LRPS
surrogate ships as a **research preview** (no accuracy claims — see README);
CUDA backends are **experimental** pending recurring GPU CI.

Detailed entries (development log, in original wording):

- **Canonicalized batch and ensemble terminology across the framework.** All Monte-Carlo-era framework names were removed (no aliases, shims, or historical re-exports remain): the propagation engine is `BatchPropagationEngine` with `BatchPropagationConfig` / `BatchPropagationResult` (`lunaris.common.batch_defs`), the backend policy moved to `lunaris.batch.backend_policy` (`BatchBackend`, `BatchBackendPlan`, `resolve_batch_backend_policy`, `BATCH_BACKEND_REQUESTS`), and the core batch kernels live in `lunaris.core.batch_propagator`. The removed `lunaris-mc` console script, the compatibility modules under `lunaris.core` and `lunaris.common`, and every `mc_*`-named config/CLI/schema field were replaced by `batch_*` (config `batch_backend`, CLI `--batch-backend` / `--batch-gravity-mode` / `--batch-dt-s` / `--batch-output-format` / `--batch-output-path`, archive fields `requested_batch_backend` / `actual_batch_backend` / `batch_backend`, progress protocol `[BATCH]` / `[BATCH_PROGRESS]` / `[BATCH_METRICS]`). Analysis reporting is ensemble-named: `EnsembleReport`, `compute_ensemble_report`, `plot_ensemble_report`, `plot_ensemble_summary`, `load_batch_result`, `build_batch_output_grid`. UI page/panel modules, docs, and tests were renamed accordingly, and a terminology hygiene test (`tests/test_terminology_hygiene.py`) bans the old names. Old artifact/output schemas are not read back — this predates any public release.
- **Made the ST-LRPS capability surface a single source of truth (Phase 0).** Added `surrogate/st_lrps/shared/capabilities.py`: a `CAPABILITIES` registry mapping every `(component, subject, feature)` triple — runtime kinds × what they predict, the baseline `(target_mode, baseline_kind)` matrix, and the two training paths — to a `CapabilityStatus` (`SUPPORTED` / `NOT_YET` / `BY_DESIGN_NA`), plus one exception type `UnsupportedCapability` (a `NotImplementedError` **subclass**, so existing `except NotImplementedError` / `pytest.raises` sites keep working). The scattered ad-hoc raises now route through it: `DirectForceRuntime`'s three scalar-potential methods raise `UnsupportedCapability` for the registry's `force_direct/scalar_residual_potential = BY_DESIGN_NA` row (a direct-force model has no `ΔU` by design). The registry renders `docs/ST_LRPS_CAPABILITY_MATRIX.md` deterministically, and a test (`tests/test_st_lrps_capabilities.py`) asserts the checked-in doc never drifts from the registry. This makes "which ST-LRPS system implements which feature, and whether a gap is a deliberate limitation or unfinished" answerable from one place.
- **Implemented the full-field spherical-harmonics baseline end-to-end (Phase 1), replacing the scattered `NotImplementedError`s.** The SH baseline potential+acceleration math (`_sh_potential_accel_batch_serial` + its Legendre table builder) was **promoted verbatim** from `st_lrps/data/spatial_cloud_generator.py` down into `lunaris.physics.spherical_harmonics` (the lowest layer), so the shared scaling layer can subtract an SH baseline without importing the data layer; the generator now imports the kernel back from physics (single source of SH math, byte-identical output — all 8 import-linter contracts stay KEPT). Added `physics.sh_potential_accel_fixed(...)` (batch geodesy potential `V` and acceleration `a = +grad U`, with a `(degree_min, degree_max]` band so one call serves both full-field evaluation and residual-baseline subtraction) and `GravityModel.potential_fixed` / `potential_accel_fixed` (which also fills the `potential_fixed` API the field-validation runner already expected). `shared/scaling.compute_base_potential_from_contract` / `compute_base_accel_from_contract` now evaluate the real SH field through `base_degree` for a full-field `spherical_harmonics` contract (the residual / point-mass / none paths stay byte-identical), threading an optional `gravity_model` through `fit_scaler_streaming`, `SobolevLoss`, the evaluation `predict_u_and_a`, and `force_direct_cli._target_accel`; the training engine loads the source model once from `meta.gravity_model_path` only when a full-field SH baseline is actually requested (the common residual path never touches the gravity file). **Validation** (astrodynamics-validation gate): potential matches the independent field oracle to ~1e-16 rel; acceleration matches the trusted `GravityModel.accel_fixed` recurrence to ~1e-16 rel off-pole (the analytic polar value matches the reference kernel; the only finite-difference disagreement is the oracle's own numerical stencil at the exact coordinate-singularity pole); the degree-0 limit reduces to the analytical monopole. Tests in `tests/test_st_lrps_sh_baseline.py`; the generator phase-parity, gravity-reference, real-CPU training-integration, and resume gates stay green.
- **Added hardware-independent compute accounting to ST-LRPS training, so "how much compute this architecture does" is portable across machines.** Wall-clock time ("training took 5 hours") is not comparable — the same run is 2 hours on one GPU and 10 on another — so the new `surrogate/st_lrps/training/compute_accounting.py` reports the field-standard unit instead: total training **FLOP** and its time-scaled form **petaflop/s-days** (1 PF-day = `8.64e19` FLOP, the OpenAI "AI and Compute" unit). GPU-hours are deliberately *not* used as the headline, because a GPU-hour is itself hardware-dependent. FLOP counts are **measured** on the real network with `torch.utils.flop_counter.FlopCounterMode` (not estimated from a heuristic), which captures the model-kind difference automatically: a `potential_autograd` model's acceleration eval and training step include the `∇ΔU` autograd (and double-backward) passes, while a `force_direct` model uses a single forward. The module reports `inference_flops_per_eval` (the portable per-acceleration "model speed", directly comparable across surrogates and to a classical SH eval), `train_step_flops_per_sample`, and `total_training_flops = train_step_flops_per_sample × total_samples_processed`, with machine-dependent context (wall-clock seconds, device name, achieved FLOP/s) quarantined under a separate `hardware` block so it never contaminates the portable numbers. The training engine accumulates the exact number of samples pushed through forward/backward and writes a `compute_accounting` block (schema `st_lrps_compute_accounting_v1`) into the run manifest; the measurement runs after checkpoints are saved and is wrapped so a measurement hiccup can never fail a real training run. The FLOP convention (a multiply-add = 2 FLOP) is pinned against a hand-computed analytic count on a Linear-only MLP. Tests in `tests/test_st_lrps_compute_accounting.py` (11) and an end-to-end manifest assertion in `tests/test_training_dataset_integration.py`.
- **Added an analytic FLOP-per-eval estimate for the classical SH gravity kernel, so the surrogate's *measured* per-evaluation cost can be compared against "the SH degree it replaces" on the same hardware-independent axis.** The numba SH kernel is opaque to `FlopCounterMode`, so `classical_sh_flops_per_eval(degree)` estimates it analytically, splitting a *provably exact* part from a *documented modelling* part: the harmonic-term count is exact — `classical_sh_terms(N) = (N+4)(N-1)/2`, matching the kernel's `n=2..N, m=0..n` double loop in `physics/spherical_harmonics.py` — while the per-term FLOP constant (`CLASSICAL_SH_FLOPS_PER_TERM ≈ 54`, with a per-term breakdown of the ALF recurrence, derivative recurrence, normalization, and acceleration/Kahan assembly documented in the source) is an order-accurate modelling assumption, exposed as an overridable parameter. The estimate scales as `O(N²)` and is explicitly labelled an *arithmetic-work* comparison only (it ignores memory traffic, latency, and GPU batch parallelism). Concretely it shows the honest picture: a hidden=256/depth=6 SIREN surrogate (~330k params) costs ~1.97 MFLOP/eval, ~5× *more* per single evaluation than classical SH degree-120 (~0.4 MFLOP) — the surrogate's advantage, if any, is degree-independent cost and batched GPU throughput, not per-eval arithmetic. Tests in `tests/test_st_lrps_compute_accounting.py` (the term count is cross-checked against a brute-force double-loop for degrees 0–129, plus `O(N²)` scaling, monotonicity, and base-overhead checks).
- **Added a surrogate-vs-classical-SH per-eval cost comparison that models the *deployed* path honestly rather than the misleading "surrogate vs full SH".** `compare_eval_cost(surrogate_inference_flops_per_eval, target_sh_degree, baseline_sh_degree)` combines the surrogate's **measured** per-eval FLOP with the **analytic** SH estimates and reports the real deployed cost — a `potential_autograd` surrogate models a residual `ΔU` on top of a low-degree analytical baseline, so the deployed path is `SH(baseline_degree) + surrogate`, and it only wins arithmetically when that sum is below the full `SH(target_degree)` field. The returned dict carries `surrogate_path_flops_per_eval`, `ratio_vs_target`, `speedup_vs_target` (>1 means less arithmetic per eval than the full classical field), the residual term band modelled, and explicit `*_source` labels (`measured` vs `analytic_estimate`) plus a `comparison_basis="arithmetic_work_only"` caveat (it ignores memory, latency, and GPU batch parallelism). For the realistic ~330k-param SIREN replacing a degree-120 field on a degree-8 baseline, `speedup_vs_target ≈ 0.2` (the surrogate path does ~5× *more* arithmetic per eval) — an honest result that locates the surrogate's value in degree-independent cost and batched GPU throughput, not single-eval FLOP. Tests in `tests/test_st_lrps_compute_accounting.py` (composition/reciprocity, a cheap-surrogate-beats-high-degree case, the realistic costlier case, and the baseline>target / non-positive-surrogate guards).
- **Surfaced the hardware-independent per-eval FLOP comparison into the gravity benchmark summary, alongside (not replacing) the existing machine-dependent wall-clock runtime.** A pure `build_compute_speed_section(...)` puts every benchmarked model on one portable axis — classical SH costs from the analytic estimate, the surrogate cost from its training run manifest's `compute_accounting` block (measured) — and the benchmark's canonical `gpu_batch_summary` now carries an additive `compute_speed` key with per-model `flops_per_eval` (+ `source` labels), the deployed-path `surrogate_vs_target` comparison, and an explicit `arithmetic_work_only` caveat. The surrogate's target/baseline degrees are read from the manifest's `resolved_config_summary` (`degree_max`/`degree_min`), falling back to the highest benchmarked SH degree. Wiring is best-effort (`_maybe_build_compute_speed` returns `None` on any problem) so it can never break a benchmark run, and the addition is backward-compatible (an extra summary key; the per-scenario CSV contracts are untouched). Tests in `tests/test_st_lrps_compute_accounting.py` (SH-only, surrogate-with-target, and surrogate-without-target sections); the benchmark summary and contract-compatibility suites stay green.
- **Surfaced the compute & speed accounting into the evaluation report, and added an optional model-FLOPs-utilization (MFU) figure backed by a sourced device-peak table.** The evaluation report (`_finalize_evaluation_report`) now carries an additive `compute_and_speed` block — the training run's measured `compute_accounting` plus a surrogate-vs-SH `compare_eval_cost` at the model's own target/baseline degree — and writes a rendered Markdown `compute_report.md` via the new pure `render_compute_report(...)` (which leads with the portable PF-days / FLOP-per-eval numbers and keeps the machine-dependent context on a clearly-labelled sub-line). Wiring is best-effort so it can never fail an evaluation. MFU: `build_compute_accounting` now derives an *optional* `model_flops_utilization = achieved_FLOP/s ÷ device_peak` when the device is recognised in `DEVICE_PEAK_FP32_FLOPS` (a small table of **theoretical FP32 non-tensor-core** peaks from NVIDIA datasheets / TechPowerUp — GTX 1660 Ti, 2080 Ti, T4, V100, A100, RTX 3090/4090, H100); an unknown device yields `None` rather than a guessed peak. The figure is documented as a conservative end-to-end lower bound (matmul-only FLOP; includes data-loading/checkpoint overhead; compared against the FP32 peak, so a TF32/AMP run reads low). Tests in `tests/test_st_lrps_compute_accounting.py` (device-peak lookup incl. unknown→None, MFU computed/None cases, and the Markdown renderer full + empty paths); 29 tests total in the file.
- **Closed a validation gap on the GPU/ST-LRPS torch SH path.** The torch spherical-harmonic evaluator (`torch_spherical_harmonics.TorchSHGravityEvaluator`, used by the GPU propagator and the gravity benchmark) shares the engine's recurrence tables but has an independent evaluation/assembly loop that was only checked for *finiteness* on tesseral fields and for *exactness* on a zero-coefficient (point-mass) field — i.e. it had the same zonal/finite-only blind spot that originally hid the Condon–Shortley phase bug. Added `test_torch_evaluator_matches_numba_engine_on_tesseral_field` (degrees 8/60/120/200): it pins the torch acceleration to the pyshtools-validated numba `GravityModel.accel_fixed` on a random field **with odd-order terms**, to ~machine precision (~3e-16), with a sanity assert that odd-`m` terms are actually present. (Noted in passing: at the *exact* pole `x=y=0` the numba and torch codes regularize the longitude singularity differently — numba zeros the horizontal components, torch returns a small finite value; this is a coordinate-singularity artifact, not a convention difference, and the parity test uses off-axis points as real orbits never sit on the axis.)
- **Scaled the field validation from 8 points to a statistical benchmark, and added model-free frame invariants.** Added `grail_degree120_pyshtools_sobol`: a deterministic ~2000-point Sobol cloud over the sphere and across 50–2000 km altitude strata (plus explicit polar/equatorial sets), generated against the independent pyshtools oracle. The field metrics now include max/P95/P99 and **latitude/altitude-binned acceleration-error tables** (`field_metrics.latitude_altitude_error_tables`); Lunaris matches pyshtools to ~machine precision in every band (worst ~8e-11 m/s² at low-altitude poles, the expected `1/cos φ` regime, far inside tolerance). Both GRAIL field benchmarks are relabeled `reference_class="independent_high_precision_field_oracle"` (the values are independent pyshtools computations, not NASA-published vectors). Added `tests/test_sh_frame_invariants.py` — kernel-free body-frame identities (z-rotation equivariance of a zonal field, zero longitudinal acceleration, pure-radial inward monopole) that catch frame transposition/sign defects. Documented, honestly, that the two stronger references remain environment-blocked and are not faked: the independent integrator needs tudatpy (not installed), and a physical rotating `MOON_PA` reference needs NAIF leapseconds (`.tls`) + lunar frame (`.tf`) kernels that are absent here (only `de440.bsp`/`de440s.bsp`/`moon_pa_de440_*.bpc` ship); the rotating-frame manifest path already fails closed rather than approximating.
- **Hardened the propagator and the trajectory-validation contract, and added an independent-cross-validation tier.** Propagator: per-chunk checkpoints now record the *end* of the completed chunk (the running state is advanced before the checkpoint write, not after, so `latest/state/last` checkpoints are valid resume points); the single-chunk path sets `stopped_early` consistently with a terminal-event stop (`sol.status == 1`) so `stop_reason` is never reported while `stopped_early` is False; and JSON telemetry is now **opt-in** (`cfg.enable_telemetry` or a positive `telem_cadence_s`) instead of always-on, so library/validation/batch callers no longer have stdout polluted. Trajectory validation runner: replaced the silent linear interpolation (mislabeled "cubic") with a **fail-closed** contract check — the reference epoch grid must be uniform and match the manifest `output_step_s`/`duration_s`, and `comparison_frame` must equal `state_frame`, else the run returns `INCOMPLETE_CONTRACT` instead of a misleading PASS/FAIL. Renamed the trajectory reference frames to `NONROTATING_FROZEN_BODY_FIXED` and corrected the "independent integrator" wording to "independent force model (pyshtools) with a separately-configured DOP853 integration" (both runs use SciPy DOP853; the integrator is **not** independent). Added a gated industry-grade cross-check `validation/independent/tudatpy_reference.py` (TU Delft Tudat, C++ core) behind a new `requires_tudatpy` marker — scaffolded and skipped cleanly when tudatpy is absent, and written to fail loudly on an API mismatch rather than return wrong numbers. Documented the multi-reference cross-validation strategy and the (firm) absence of any public NASA gravity-only truth trajectory in `docs/GRAVITY_ENGINE_EXTERNAL_VALIDATION.md`.
- **Fixed a Condon–Shortley phase bug in the spherical-harmonic kernel.** `build_legendre_coeffs` was multiplying the per-order `scale_m` table by `(-1)^m`, applying the Condon–Shortley phase to fully-normalized GRAIL/geodesy coefficients that are defined *without* it. This flipped the sign of every odd-order (tesseral/sectoral) term, producing accelerations ~1e-3 m/s² wrong on the real GRAIL degree-120 field — invisible to the J2/zonal tests (m=0 is unaffected) but corrupting any orbit driven by the tesseral field. Removed the phase factor; `accel_fixed` now matches **pyshtools** (`MakeGravGridPoint`, which excludes the phase) to ~1.9e-13 m/s² and the independent oracle to ~1e-12. Also corrected the two validation harnesses that had been "aligned" to the wrong convention: `validation/independent/independent_sh.py` (negative sectoral seed → positive, no phase) and `validation/independent/pyshtools_reference.py` (removed the manual `(-1)^m` coefficient flip; pyshtools is called with its no-phase default). Net result: engine, both independent harnesses, the degree-4 oracle, pyshtools, and the GRAIL convention now all agree on **no Condon–Shortley phase**. Tests in `tests/test_independent_sh_validation.py`.
- **Propagated the Condon–Shortley fix into the ST-LRPS data path and gated stale artifacts.** The spatial-cloud generator (`spatial_cloud_generator._sh_potential_accel_batch_serial`) had kept the old negative sectoral recurrence (`P_n[1] = -√3·c`, `P_n[n] = -diag·c·P_{n-1}`), i.e. the `(-1)^m` phase — so it produced training labels on a sign-flipped odd-order field while the corrected runtime uses the no-phase field. Flipped the sectoral seed/recurrence to positive (the first sub-diagonal and vertical recurrence keep `m` fixed and were already phase-agnostic); the generator now matches `GravityModel.accel_fixed` to ~4e-16 m/s² on a field with odd-`m` tesseral/sectoral terms. Added `tests/test_st_lrps_generator_phase_parity.py` — a generator-vs-engine acceleration parity test that deliberately exercises odd-`m` coefficients (a zonal-only check cannot catch a phase error) plus a negative control proving the test detects a Condon–Shortley field. Stamped a new `spherical_harmonic_convention="4pi_geodesy_no_condon_shortley_v1"` (+ `gravity_label_engine_version="lunaris_sh_v2"`) into dataset attributes, the `DatasetContract` (validated; defaults to `None` so pre-fix datasets fail closed unless an explicit legacy override), and the artifact dataset-meta block. Marked the accuracy results in `README.md` and `docs/BENCHMARK_RESULTS.md` as **invalidated/pending regeneration** (they predate the phase fix; datasets must be regenerated, the model retrained, and benchmarks re-run — only relative runtime figures stay indicative; old checkpoints are incompatible with the corrected runtime).
- Reworked the external gravity-reference validation so it tests whether the Lunaris engine is *actually* right, and removed misleading benchmarks. **Field:** the primary high-degree truth is now the GRAIL JGGRX_1800F degree-120 **pyshtools** benchmark, which Lunaris matches to machine precision (potential rel ~1.8e-16, acceleration ~1.9e-13 m/s²); the synthetic degree-4 independent-oracle benchmark is retained as a low-degree from-scratch cross-check (~1e-12). The degree-120 *internal direct-formula oracle* benchmark was deleted: its classical associated-Legendre recurrence loses ~8 digits above degree ~80, so it disagreed with both Lunaris and pyshtools (~1e-5) and was failing the engine for the reference's own error. **Trajectory:** the previously fail-closed layer now runs a real gravity-only comparison — `lunaris_trajectory_runner` propagates the manifest initial state with the production propagator (`core.propagator.propagate` over `DynamicsEngine`, `enable_sh` + `allow_identity_rotation`) and compares epoch-by-epoch against an immutable reference arc generated by an **independent** library (pyshtools force) and integrator (SciPy `DOP853` at tighter tolerances). On the committed GRAIL degree-32 ~2-orbit arc, Lunaris reproduces the independent reference to ~5e-8 m max position error and conserves the gravity-only Hamiltonian to ~4e-15 (added `specific_energy_drift` as an internal truth check). A rotating body-fixed field (`state_frame != gravity_fixed_frame`) fails closed with `INCOMPLETE_CONTRACT` (SPICE lunar orientation is out of scope). Honesty note in the docs: NASA/JPL publishes no gravity-only truth trajectory (all mission products carry full force models + orbit determination), so this is an independent-tool-generated reference, not NASA-published truth. Deleted stale committed run outputs (`out_grail/`, `out_pysh/`, `coverage_output.txt`, `runs.json`) and git-ignored them. Tests in `tests/test_gravity_reference_runner.py` (real trajectory PASS + rotating-frame rejection) and `tests/test_gravity_reference_oracle_metrics.py` (energy-drift).
- Added a run-level training preset `--run-preset {custom,development,quick,paper}` (orthogonal to the architecture-only `--model-preset`). `custom` (default) is a no-op and fully backward-compatible; `development` uses the fast seeded_random interpolation split; `quick` runs the pipeline smoke path; `paper` enforces a scientifically defensible posture — a generalization split (`spatial_block` by default; explicit interpolation splits raise), deterministic execution, AMP/benchmark-cudnn/quick/legacy shortcuts off, and an enforced preflight gate — with a hard error on any conflicting explicit flag. Wired into `config.parse_args` (before the model preset) and `engine.train` (before `set_seed`, idempotent re-application); `run_preset`/`split_policy` recorded in the config summary, run manifest, and `config.json`/`training_config_hash`. Tests in `tests/test_st_lrps_run_preset.py`.
- Added independent external-reference validation harnesses under `validation/independent/` to break the correlated-error problem (the in-tree gravity validation reused Lunaris' own code). `independent_sh.py` evaluates the geopotential with a self-contained classical associated-Legendre recurrence (no `scipy.special` dependency, so it is version-proof) + an explicit 4π normalization, and takes the acceleration as the numerical gradient of that potential (a different algorithm from the analytic Numba recurrence under test); it matches Lunaris to ~1e-12 on synthetic models and the real lunar model. `naif_ephemeris.py` cross-checks `EphemerisManager` against direct `spiceypy.spkpos` queries. `pyshtools_reference.py` adds an optional second SH opinion behind the new `requires_pyshtools` pytest marker (skipped cleanly when absent). Tests in `tests/test_independent_sh_validation.py`; validation docs updated. Set `[tool.pytest.ini_options] pythonpath = ["."]` so the top-level `validation/` harness package (intentionally not installed) imports under a bare `pytest` invocation as used in CI.
- Packaging metadata: set the project author/maintainer to Ayberk Demirkanat (`pyproject.toml`, `CITATION.cff`, `LICENSE`), added `[project.urls]` (homepage/repository/issues/changelog), and added `CONTRIBUTING.md` (dev setup + `ruff`/`mypy`/`lint-imports`/pytest quality gates) and `SECURITY.md` (private vulnerability reporting, untrusted-artifact scope). Metadata-only; dependency locks are unaffected.
- Slimmed the README from 615 to ~280 lines into a landing page: a Documentation table as the navigation hub, condensed force-model/install/ST-LRPS/resume sections that defer to the canonical `docs/`, and a 3-row benchmark highlight table linking the full tables in `docs/BENCHMARK_RESULTS.md`. Content only; no code or output-format changes.
- Began the `engine.train()` decomposition by extracting the best-checkpoint concern into a unit-tested `training/checkpoint_manager.py`: `BestCheckpointTracker` (owns `best_val`/`best_epoch`/`epochs_without_improve`; strict-`<` best, burn-in gate, patience, resume restore, early-stop) and `resolve_best_ckpt_start_epoch()` (the auto/manual burn-in start-epoch policy). The training loop delegates the decision to the tracker and mirrors its state into the existing local names, so all downstream payload/manifest/summary read-sites are unchanged. No behavior change (verified against the real-training integration and resume tests); tests in `tests/test_st_lrps_checkpoint_manager.py`.
- Continued the `engine.train()` decomposition by extracting its three most self-contained setup phases into module-level helpers, each moved **verbatim** with no logic change: phase 4 (data splitting) → `_resolve_data_splits()` returning a `_SplitResult`; phase 7 (residual scaler fit) → `_fit_residual_scalers()` (the train-only `fit_scope` leakage-safe contract and provenance are preserved exactly); phase 8 (DataLoader construction) → `_build_dataloaders()` returning a `_LoaderResult` (the `worker_init_fn` seeding and the explicit per-epoch shuffle `torch.Generator` determinism are preserved exactly). `train()` calls each helper and unpacks the results into the existing local names, so every downstream read-site is unchanged. No behavior change (verified against the real-training integration and resume tests).
- Completed the `engine.train()` / `evaluate()` decomposition (reviewer item #3). Extracted the two remaining setup phases verbatim — phase 2-3 (dataset discovery, metadata SSOT, contract load + validation) → `_load_dataset_context()` returning a `_DatasetContext`, and phase 9 (model/GradNorm/Sobolev-loss/AdamW build, including the reload-safety degree/band resolution and resume restores) → `_build_model_and_optim()` returning a `_ModelBundle`. Then split `train()` itself into `build_training_session()` (setup phases 1-10, returning a `_TrainingSession` dataclass whose fields are exactly the 28-local boundary that crosses into the loop, determined by static analysis) and `_run_training_loop()` (phase 11, which unpacks the session into the original local names so the loop body is byte-for-byte unchanged); `train()` is now a two-line orchestrator. On the evaluation side, the report-assembly tail of `evaluate()` (build report dict, write `eval_report.json`/`evaluate_metrics.json`, emit CSVs/summary/manifest) moved verbatim into `_finalize_evaluation_report()`. No behavior change — verified after each step with ruff, the import-linter contracts, the real-training integration + resume tests, and the end-to-end `evaluate()` tests.
- Standardized all generated artifacts under a single, git-ignored `outputs/` root with typed subdirectories (`simulations/`, `monte_carlo/`, `missions/`, `gravity_benchmark/`, `training/`, `evaluations/`, `runtime/`, `dataset_reports/`, `datasets/`, `visualization/`). Updated every default output path to match.
- Reworked the ST-LRPS Studio "Gravity Plots" page into a folder-first workflow: pick a results folder, then choose from the models discovered inside it.
- Aligned documentation and configuration (`README.md`, `docs/`, `validation/` docs, `pyproject.toml`, `environment.yml`, `CITATION.cff`) with the unified `lunaris` naming and removed legacy output-directory references.
- Removed the root-level compatibility launchers (`main.py`, `ui.py`, `studio.py`, `mc_runner.py`); use the installed console entry points (`lunaris`, `lunaris-ui`, `lunaris-studio`, `lunaris-mc`) or `python -m` instead. Updated docs and Slurm examples accordingly. The desktop UI now launches the backend from the installed package module files (`lunaris/cli/main.py`, `lunaris/core/mc_runner.py`, resolved relative to the `lunaris` package) instead of the deleted root scripts; `ui_commons` project-root discovery no longer keys on `main.py`.
- Unified naming: project-root discovery now keys on `pyproject.toml`/`.git`/`data/assets` instead of the folder name; loader environment overrides renamed `LUNARSIM_*` → `LUNARIS_*` (generic `LDEM_ROOT`/`ALBEDO_ROOT`/`SPICE_KERNELS` fallbacks retained); fixed the misspelled `topografy_models` data directory to `topography_models`.
- Removed dead legacy code: the deprecated `get_provider()` surface-provider alias (use `as_numba_dict()`) and the `enable_solid_tides` perturbation-flag alias (use `enable_tides_k2`/`enable_tides_k3`). Backward-compatible loaders for old checkpoints/sessions/datasets and report metadata are preserved.
- Restructured documentation around a single canonical architecture reference (`docs/ARCHITECTURE.md`): slimmed the README "Repository Architecture" section to a quick map plus a pointer, added a top-level documentation index, switched the install instructions to editable installs (`pip install -e .` / `.[all]`), and trimmed the duplicated architecture block from `CLAUDE.md` so it summarizes and links the canonical doc instead of drifting out of sync.
- Removed remaining prototype-era naming so the project reads as a single finished structure: updated stale `models.*` module-path references in docstrings/comments to `lunaris.physics.*` (`core/dynamics.py`, `core/config.py`, `loaders/spice_builder.py`, `physics/gravity_adapter.py`, `cli/main.py`); dropped dead `lunar_simulation.models.*` fallback imports in `tests/test_solar_effects.py` and `tests/test_relativity_effects.py`; and stripped development-iteration version labels (e.g. "Overhauled v7", "ENHANCED v7", "UPDATED v12", `studio - v3` headers, `_overhauled` method suffixes). Functional on-disk schema versions (checkpoint/manifest/derivative-convention) are retained.
- Deleted dead code: removed fully-shadowed duplicate definitions in the reporting layer (`figure_run_config_page` in `analysis/reporting/manager.py`; `draw_table`, `draw_kv_block`, `draw_kv_table`, `metrics_rows` in `analysis/reporting/plotting.py`) — the earlier definitions were superseded by later same-name definitions and never executed (~370 lines). Also pruned unused stdlib/typing imports from leaf modules. Re-export barrels (`ui/widgets/force_models_page.py`, `surrogate/st_lrps/ui/studio_parts/*`) and the `qt_common` PySide6/PyQt6 compatibility shim were intentionally left intact.
- Split the 7,645-line gravity-benchmark god-module into cohesive modules without behavior changes. `surrogate/st_lrps/evaluation/compare_gravity_models.py` is now a thin CLI/orchestrator (~495 lines, keeping `main`, argparse, and the module path the Studio UI launches); the implementation moved into a new internal subpackage `surrogate/st_lrps/evaluation/_gravity_benchmark/` (`types`, `compute`, `metrics`, `plotting`, `results_io`, `modes`). `compare_gravity_models` re-exports every implementation symbol as a backward-compatible facade, so `compare_gravity_models.X` / `from ...compare_gravity_models import X` keep working unchanged. The module path, `lunaris-benchmark` entry point, CLI flags, and outputs are unchanged. (Tests that monkeypatch an internal helper now target its canonical module `_gravity_benchmark.compute`.)
- Tamed the two largest functions by extracting their output-only sections into helpers, with **no logic change** (each extracted block is byte-for-byte identical to the original and provably has no state flowing back out): `evaluate()` (`surrogate/st_lrps/evaluation/cli.py`) 1,427 → 1,066 lines via `_save_evaluation_plots` / `_write_evaluation_csvs` / `_print_evaluation_summary`; `train()` (`surrogate/st_lrps/training/engine.py`) 1,657 → 1,546 lines via `_log_dataset_and_model_summary` / `_log_training_curriculum` / `_log_data_loading_policy`. The training loop, metrics computation, and all control flow are untouched.
- Replaced the stale duplicate `slurm_examples/` template directory with the canonical `hpc/` Slurm templates (`slurm_train_stlrps.sbatch`, `slurm_benchmark_gpu.sbatch`, `slurm_mc_array.sbatch`, plus the shared `env_template.sh`), removing the four tracked `slurm_examples/*.slurm` files. Repositioned `docs/HPC.md` around the primary HPC workload — headless ST-LRPS dataset/spatial-cloud generation, training, evaluation, and orbit-level gravity benchmark/validation, plus Monte Carlo — with explicit guidance to keep GUI dependencies (`lunaris-ui`/`lunaris-studio`) off compute nodes. Updated the README HPC references to `hpc/` and added the missing `lunaris-benchmark` console entry point to the README entry-point list. Documentation/template only; no runtime, schema, or output-format changes.
- Added a manifest-driven external-data acquisition CLI (`lunaris-data` = `lunaris.cli.data:main`) for gravity, SPICE/ephemeris, topography, albedo, and dataset assets, driven by `data/data_sources.json` (data root resolved as `--data-dir` → `LUNARIS_DATA_DIR` → repo `data/`; supports `list`, `download`, `verify`, `path` with atomic `.part` downloads and optional SHA-256 verification). Documented the `LUNARIS_DATA_DIR` scratch/project-storage workflow for HPC, fixed the Slurm submit examples to avoid unsafe submit-time `$SLURM_JOB_ID` expansion (use an exported `LUNARIS_OUTPUT_DIR` plus a timestamped run name), and extended `.gitignore` for external data/kernels while keeping the manifest and README tracked. Headless and GUI-free; no numerical, training, benchmark, cache, or schema changes.

## 0.1.0

- Introduced the `lunaris` package namespace and `src/` layout.
- Preserved ST-LRPS as `lunaris.surrogate.st_lrps`.
- Added console entry points for CLI, GUI, Studio, Monte Carlo, training, evaluation, and orbit benchmark workflows.
