# Baseline Technical Audit

**Lunaris Software Hardening — Phase 1, Deliverable §1**

This document is the technical-debt register produced by the Phase 1 baseline
audit. Its goal is **not** to add scientific features but to establish whether
the existing system is cleanly installable, testable, reproducible, and
auditable. No physics, ST-LRPS math, loss functions, or propagation behavior is
changed by this audit — findings that require code changes are tracked here and
implemented in separate, small, reviewable PRs.

> Companion deliverables: [`INSTALLATION_MATRIX.md`](INSTALLATION_MATRIX.md),
> [`BENCHMARK_PROVENANCE_AUDIT.md`](BENCHMARK_PROVENANCE_AUDIT.md),
> and the machine-readable `baseline_manifest.json`.

---

## 1. Audit reference environment

The audit was performed against a frozen reference state. Regression checks in
later refactors should compare against this baseline.

| Field | Value |
|---|---|
| Git commit | `94af969331bf27fe853962eeb8153c5c02b84484` |
| Branch | `feature/lunaris-updates` |
| OS | Windows 11 Home 10.0.26200 |
| Python | 3.12.1 |
| NumPy / SciPy | per installed environment (lower-bound pinned only — see DEP-04) |
| PyTorch | 2.5.1+cu121 (`torch.cuda.is_available() == True`) |
| Qt bindings | PySide6 **and** PyQt6 both installed and importable |
| CPU test collection | 1335 tests collected, 13 deselected, 0 collection errors (`-m "not slow and not requires_data and not requires_cuda"`) |

The package imports cleanly (`import lunaris`) and the CPU test suite collects
without import errors, which is a healthy starting point: the debt below is
about reproducibility, packaging hygiene, provenance labeling, and CI coverage —
not broken code.

---

## 2. Severity scale and how to read a finding

| Severity | Meaning |
|---|---|
| **Critical** | Can produce or mislabel scientific results, or block a documented acceptance criterion. |
| **High** | Reproducibility / install correctness risk; likely to bite in a clean environment. |
| **Medium** | Maintenance, documentation, or packaging hygiene; not blocking but accruing. |
| **Low** | Cosmetic / local clutter; safe to defer. |

Each finding records: **file/module**, **description**, **impact**,
**severity**, **proposed fix**, **target PR**, and **status** (Open · Planned ·
Done · Won't-fix). Statuses are updated as the Phase 1 PRs land.

### Summary by severity

| Severity | Open/Planned | Done |
|---|---|---|
| Critical | 1 | 0 |
| High | 5 | 0 |
| Medium | 11 | 1 |
| Low | 4 | 0 |

The single Critical finding (**SCI-01**) is the benchmark backend-labeling gap
and is the highest priority for the provenance PR.

---

## 3. Findings

### 3.1 Scientific result reliability risk

#### SCI-01 — Silent CPU fallback in the gravity-model benchmark (Critical)

- **File/module:** [`_gravity_benchmark/types.py`](../src/lunaris/surrogate/st_lrps/evaluation/_gravity_benchmark/types.py) — `GravityModelCache._load` (≈L256–305); provenance: [`evaluation/provenance.py`](../src/lunaris/surrogate/st_lrps/evaluation/provenance.py) — `build_benchmark_manifest`.
- **Description:** When ST-LRPS GPU is requested but CUDA/PyTorch is unavailable, the loader falls back to CPU and announces it only via `print(...)`. The benchmark provenance manifest captures config/model/environment/git hashes but does **not** record `requested_backend` / `actual_backend`, `requested_device` / `actual_device`, `requested_sh_degree` / `actual_sh_degree`, or `fallback_applied` / `fallback_reason`. A run that silently ran on CPU can therefore be saved and named as if it ran on GPU.
- **Impact:** Violates Phase 1 §3 and the acceptance criterion *"CPU fallback must not be labeled as a GPU result."* Benchmark provenance and naming cannot be trusted for backend/device.
- **Proposed fix:** A clean reference pattern already exists in this repo — [`core/mc_backend_policy.py`](../src/lunaris/core/mc_backend_policy.py) `MCBackendPlan` already tracks `requested_backend`/`actual_backend`/`requested_sh_degree`/`actual_sh_degree`/`fallback_reason` and refuses silent fallback. Extend the benchmark manifest with the same `requested_*`/`actual_*` + `fallback_applied`/`fallback_reason` fields, surface fallback as an explicit warning (not just `print`), and make the benchmark name/label reflect the **actual** backend.
- **Target PR:** #3 (Benchmark provenance). **Status:** Open.

#### SCI-02 — `actual_degree ≤ backend capability` not asserted in benchmark provenance (High)

- **File/module:** `evaluation/provenance.py`; cf. `mc_backend_policy._gpu_sh_limits` and `GPU_SH_MAX_DEGREE`.
- **Description:** The MC path correctly routes a requested SH degree above the true GPU kernel limit (degree ≤ 24) to an explicit CPU fallback **without clipping** ([`mc_backend_policy.py:482`](../src/lunaris/core/mc_backend_policy.py)). The benchmark provenance manifest does not record backend capability or assert `actual_sh_degree ≤ capability`, so the same guarantee is not auditable for benchmark runs.
- **Impact:** A high-degree "GPU" benchmark could be reported without evidence that the degree was actually achievable on the GPU kernel.
- **Proposed fix:** Record `gpu_sh_max_degree` / supported tiers in the benchmark manifest and add a consistency assertion `actual_sh_degree ≤ gpu_sh_max_degree` for GPU-labeled runs.
- **Target PR:** #3. **Status:** Open.

#### SCI-03 — Output statistical-consistency checks exist but are not wired into MC reports (Medium)

- **File/module:** [`tests/test_benchmark_validation.py`](../tests/test_benchmark_validation.py), `evaluation/benchmark_validation.py` (`validate_benchmark_outputs`).
- **Description:** *Good news first:* the gravity benchmark already has a well-tested output validator covering most of §3's required checks — finite metric values, metric ordering (`median ≤ p95 ≤ max`), `runtime > 0`, sample-count consistency, duplicate/inconsistent model names, scenario-id contiguity. The gap is that these checks are scoped to the **benchmark** CSV outputs only; the Monte Carlo result path has no equivalent automatic gate, and the `sample_count > 0` / `actual_degree ≤ capability` checks from §3 are not yet part of the same validator.
- **Impact:** Inconsistent statistics could pass silently for non-benchmark result flows.
- **Proposed fix:** Reuse `validate_benchmark_outputs` building blocks for MC summaries; add the two missing assertions (`sample_count > 0`, `actual_degree ≤ capability`). Make report generation raise/warn explicitly on inconsistency (it already does for benchmarks).
- **Target PR:** #3. **Status:** Open.

#### SCI-04 — Lower-bound-only numeric dependencies threaten benchmark reproducibility (High)

- **File/module:** [`pyproject.toml`](../pyproject.toml) `dependencies` / extras.
- **Description:** All numeric deps are pinned with lower bounds only (`numpy>=1.20`, `scipy>=1.8`, `torch>=2.0`, …). Two clean installs months apart can resolve materially different NumPy/SciPy/Torch versions, which can move benchmark numbers without any code change.
- **Impact:** "Reproducible benchmark" claims are not enforceable across environments. Directly relevant to the §8 criterion *"no unexplained regression in physical results."*
- **Proposed fix:** Add a `constraints.txt` / lock per install profile capturing exact resolved versions; document its use. Keep `pyproject` lower bounds as the compatibility floor. (See DEP-04.)
- **Target PR:** #1 (Dependencies). **Status:** Open.

### 3.2 Packaging and dependency issues

#### DEP-01 — Install extras do not match the six required profiles (High)

- **File/module:** [`pyproject.toml`](../pyproject.toml) `[project.optional-dependencies]`.
- **Description:** Current extras are `core / ml / hpc / ui / reports / dev / all`. Phase 1 §4 requires the profiles `core-cpu`, `ml-cpu`, `ml-cuda`, `hpc-headless`, `desktop-ui`, `development`. There is no CPU/CUDA split and no headless-vs-desktop split.
- **Impact:** A clean install cannot select a reproducible, intent-matched profile; HPC and CPU users get the wrong dependency closure.
- **Proposed fix:** Redefine extras to the six named profiles; keep backwards-compatible aliases if needed.
- **Target PR:** #1. **Status:** Open.

#### DEP-02 — `ui` extra bundles both PySide6 and PyQt6 (Medium)

- **File/module:** [`pyproject.toml`](../pyproject.toml) `ui` / `all` extras (both installed in the reference env).
- **Description:** The desktop UI extra requires **both** Qt bindings. The application entry point is PySide6-based (`lunaris.ui.app`); pulling PyQt6 as well roughly doubles the Qt footprint and risks binding conflicts.
- **Impact:** Install bloat and potential `PySide6`/`PyQt6` runtime conflicts; unnecessary for headless installs.
- **Proposed fix:** Audit actual `PyQt6` imports; standardize on one binding (PySide6) and drop or isolate the other behind a separate optional extra.
- **Target PR:** #1. **Status:** Open (needs a quick import census first).

#### DEP-03 — `ml` and `hpc` extras are identical and not headless-clean (Medium)

- **File/module:** [`pyproject.toml`](../pyproject.toml).
- **Description:** `ml` and `hpc` both resolve to exactly `torch + h5py`. `hpc` therefore does not differ from `ml`, and neither guarantees a Qt-free / Node-free headless closure as its name implies.
- **Impact:** "hpc-headless" cannot be selected as a distinct, Qt-free profile.
- **Proposed fix:** Define `hpc-headless` explicitly with no Qt/Node deps; differentiate from `ml-cpu`/`ml-cuda`.
- **Target PR:** #1. **Status:** Open.

#### DEP-04 — No lock/constraints for reproducible resolution (High)

- **File/module:** repo root (none present).
- **Description:** No `constraints.txt`, lockfile, or pinned environment is produced for any profile.
- **Impact:** Reproducible setup (a Phase 1 goal) is not achievable; pairs with SCI-04.
- **Proposed fix:** Generate per-profile constraints from a known-good resolve; document refresh procedure. Verify a clean venv install matches.
- **Target PR:** #1. **Status:** Open.

#### DEP-05 — Four parallel dependency declarations drift (High)

- **File/module:** [`pyproject.toml`](../pyproject.toml), [`requirements.txt`](../requirements.txt), [`requirements_hpc.txt`](../requirements_hpc.txt), [`environment.yml`](../environment.yml).
- **Description:** Dependencies are declared in four places. `requirements.txt` force-installs `torch + PySide6 + PyQt6` (heavy, CUDA + dual-Qt) regardless of use case and is hand-maintained, so it diverges from the `pyproject` extras that CLAUDE.md designates as the SSOT.
- **Impact:** Contradictory install instructions; CPU/headless users pulled into heavy CUDA+Qt closures; doc/reality mismatch.
- **Proposed fix:** Make `pyproject.toml` the single source; auto-generate any `requirements*.txt` from it (or delete in favor of profiles + constraints); reconcile or remove `environment.yml`.
- **Target PR:** #1. **Status:** Open.

#### DEP-06 — CPU install cannot avoid CUDA wheels by default (Medium)

- **File/module:** [`pyproject.toml`](../pyproject.toml) `ml`/`hpc`; CI installs torch via the CPU index explicitly ([`.github/workflows/tests.yml:53`](../.github/workflows/tests.yml)).
- **Description:** `torch>=2.0` with default index resolves to CUDA builds on Linux. Only CI works around this with `--index-url .../whl/cpu`; ordinary users have no documented CPU path.
- **Impact:** CPU-only users download multi-GB CUDA wheels; contradicts §4 *"CUDA packages must not be mandatory for a CPU install."*
- **Proposed fix:** Document the CPU torch index per OS in the install matrix; encode it in `ml-cpu`/`hpc-headless` guidance.
- **Target PR:** #1. **Status:** Open.

### 3.3 Installation and deployment risk

#### INST-01 — Headless install not isolated from Qt/Node (Medium)

- **File/module:** [`pyproject.toml`](../pyproject.toml) extras; web app under `src/lunaris/ui/web/`.
- **Description:** There is no profile that guarantees a Qt-free, Node-free headless closure (see DEP-03). The web 3D preview source lives **inside** the importable package tree.
- **Impact:** Headless HPC nodes risk pulling, or being asked to build, UI/Node dependencies.
- **Proposed fix:** Ship a verified `hpc-headless` extra and document that core/HPC imports require neither Qt nor Node. Add a smoke test asserting `import lunaris.core.*` works without Qt installed.
- **Target PR:** #1 (+ smoke test in #5). **Status:** Open.

#### INST-02 — Next.js web app committed inside the Python source package (Medium)

- **File/module:** [`src/lunaris/ui/web/package.json`](../src/lunaris/ui/web/package.json) (Next 16, React 19, three.js, @react-three/*), `next.config.ts`, app sources, tracked `*_real.webp` textures.
- **Description:** A full Next.js/React-Three-Fiber app is version-controlled within `src/lunaris/ui/web/`. `node_modules/`, `.next/`, and heavy textures are correctly gitignored, but the JS/TS source + selected textures are tracked inside the Python package directory.
- **Impact:** Couples a Node toolchain to the Python source tree; risk that sdist/wheel packaging includes web assets unless explicitly excluded; npm dependency surface (Next 16 / React 19) is unscanned.
- **Proposed fix:** Confirm the web dir is excluded from sdist/wheel (it is not a Python package, but verify `MANIFEST.in`/package-data). Treat the web app as an optional component with its own README; add npm audit in CI (see CI-04).
- **Target PR:** #1 packaging check + #2 CI. **Status:** Open.

#### INST-03 — External scientific data resolution is documented but unverified in CI (Medium)

- **File/module:** [`data/README.md`](../data/README.md), [`data/data_sources.json`](../data/data_sources.json), `lunaris-data` CLI; `.gitignore` excludes all large rasters/kernels.
- **Description:** Large inputs (SPICE kernels, gravity/topography/albedo rasters) are correctly *not* bundled and fetched via `lunaris-data`. Data-dependent tests self-skip when inputs are absent. There is no CI assertion that the data-download/verify contract still resolves (it would require network/large data, reasonably excluded from PR CI).
- **Impact:** Silent rot of the data contract would only surface on a real run.
- **Proposed fix:** Keep heavy data out of PR CI; add a periodic/manual job that runs `lunaris-data verify` against the manifest. Capture the data manifest hash in `baseline_manifest.json`.
- **Target PR:** #5 (baseline) + periodic CI in #2. **Status:** Open.

### 3.4 CI quality gates (test infrastructure)

#### CI-01 — Python version matrix missing 3.12 (Medium)

- **File/module:** [`.github/workflows/tests.yml:28`](../.github/workflows/tests.yml).
- **Description:** CI matrix is `["3.10", "3.11"]`. Phase 1 §5 requires 3.10–3.12, and the reference dev environment is already 3.12.1.
- **Impact:** The version the maintainer actually uses is untested in CI.
- **Proposed fix:** Add `"3.12"` to the matrix.
- **Target PR:** #2. **Status:** Open.

#### CI-02 — No lint / format / import-order / type-check gate (Medium)

- **File/module:** repo (no `ruff`, `flake8`, `mypy`, `pre-commit`, or `setup.cfg` config is tracked).
- **Description:** §5 requires Ruff-or-equivalent lint, format check, import-order, basic type checking, and unused-import / unreachable-code checks. None exist.
- **Impact:** Style/dead-code/import regressions land unguarded.
- **Proposed fix:** Add a Ruff config (lint + format + import sort + `F401`/unreachable) and a lightweight type-check job; run on the 3.x matrix.
- **Target PR:** #2. **Status:** Open.

#### CI-03 — No wheel/sdist build + clean-install + entry-point smoke (Medium)

- **File/module:** `.github/workflows/` (no packaging job).
- **Description:** §5 and §8 require building a wheel and sdist, installing the wheel into a clean environment, and verifying all console entry points. No such job exists.
- **Impact:** Packaging breakage and missing entry points are caught only by users.
- **Proposed fix:** Add a `build` job (`python -m build`) + install the wheel in a fresh venv + run `--help` on every `lunaris-*` entry point.
- **Target PR:** #2. **Status:** Open.

#### CI-04 — No security scanning (Medium)

- **File/module:** `.github/workflows/`.
- **Description:** §5 requires Python dependency vulnerability scanning, npm dependency scanning (the web app pulls Next 16 / React 19), and secret scanning. None present.
- **Impact:** Known-vulnerable deps and leaked secrets go undetected.
- **Proposed fix:** Add `pip-audit` (Python), `npm audit` scoped to `src/lunaris/ui/web`, and a secret scanner (e.g. gitleaks) as a CI job.
- **Target PR:** #2. **Status:** Open.

#### CI-05 — Single monolithic CPU test job; no test grouping or parallelism (Low)

- **File/module:** [`.github/workflows/tests.yml:62`](../.github/workflows/tests.yml).
- **Description:** §5 envisions grouped jobs (core CPU, ML CPU, headless UI, benchmark smoke, artifact contract, CLI smoke). Currently one job runs the full 1335-test suite serially. GPU/data tests are already correctly excluded from PR CI (good).
- **Impact:** Slower feedback; harder to see which subsystem failed.
- **Proposed fix:** Split into marked groups; optionally add `pytest-xdist`. Keep GPU/`requires_data` on separate manual/periodic/self-hosted runners.
- **Target PR:** #2. **Status:** Open.

### 3.5 Documentation gaps

#### DOC-01 — Phase 1 deliverable docs missing (Medium)

- **File/module:** `docs/`.
- **Description:** `BASELINE_AUDIT.md` (this file), `INSTALLATION_MATRIX.md`, and `BENCHMARK_PROVENANCE_AUDIT.md` are required §7 deliverables. This file is now created; the other two and `baseline_manifest.json` remain.
- **Impact:** Audit/provenance/install guarantees are not documented.
- **Proposed fix:** Author the remaining two docs and the manifest.
- **Target PR:** this doc lands the audit; #1/#3/#5 land the rest. **Status:** Partially done (this file).

#### DOC-02 — README install story predates the six profiles (Medium)

- **File/module:** [`README.md`](../README.md) (≈L197–248).
- **Description:** README documents `pip install -e .`, `.[all]`, `.[hpc]` — not `core-cpu`/`ml-cpu`/`ml-cuda`/`hpc-headless`/`desktop-ui`/`development`, and does not give per-OS (Windows/Ubuntu) clean-install commands required by §4/§8.
- **Impact:** Documented install path diverges from the intended reproducible profiles.
- **Proposed fix:** Update README to point at `INSTALLATION_MATRIX.md`; add Windows + Ubuntu clean-install + smoke commands.
- **Target PR:** #1. **Status:** Open.

#### DOC-03 — No machine-readable baseline manifest of reference runs (Medium)

- **File/module:** repo root (none).
- **Description:** §2/§7/§8 require frozen reference smoke runs (single-orbit propagation, small Monte Carlo, SH gravity eval, ST-LRPS artifact load+inference, field-level eval, orbit-level benchmark smoke, headless UI import, CLI `--help`) captured with commit SHA, Python/OS/dep versions, device, seed, config, runtime, key metrics, and output hashes.
- **Impact:** No regression anchor for later refactors.
- **Proposed fix:** Add reproducible smoke scenarios + a generator that writes `baseline_manifest.json`.
- **Target PR:** #5 (Baseline & smoke infra). **Status:** Open.

### 3.6 Unused / temporary development files

#### CLEAN-01 — `fix_imports.py` at repo root (Low)

- **File/module:** [`fix_imports.py`](../fix_imports.py) (tracked).
- **Description:** One-time script that rewrote `from .x` UI imports to absolute `lunaris.ui.<folder>.<mod>` paths during a past refactor. No longer referenced by any workflow.
- **Impact:** Root-level clutter; could be re-run accidentally.
- **Proposed fix:** Move to `tools/migrations/archive/` (preserve history) or delete after confirming no references.
- **Target PR:** #4 (Cleanup). **Status:** Open.

#### CLEAN-02 — `refactor_ui.py` at repo root (Low–Medium)

- **File/module:** [`refactor_ui.py`](../refactor_ui.py) (tracked).
- **Description:** One-time script that physically `os.rename`-moved UI widget files into `pages/components/core/` and rewrote imports. It is **destructive if re-run** against the current layout.
- **Impact:** Root clutter plus a real foot-gun (accidental file moves).
- **Proposed fix:** Archive to `tools/migrations/archive/` or delete; if archived, add a top-of-file guard/comment noting it is historical and must not be re-run.
- **Target PR:** #4. **Status:** Open.

#### CLEAN-03 — Local working-tree clutter (untracked) (Low)

- **File/module:** `tools/hpc/__pycache__/*.pyc`, `outputs/aiaa_scitech/generate_publication_plots*.py` (v1 + v2), `src/lunaris/ui/web/node_modules/`.
- **Description:** Present on disk but **not git-tracked** (correctly gitignored). Noted for completeness so they are not mistaken for repo debt. The two `generate_publication_plots` versions, if ever promoted, should be de-duplicated and moved under `tools/`.
- **Impact:** None on the repo; local disk only.
- **Proposed fix:** No repo action required. Optionally `git clean -ndx` review locally.
- **Target PR:** — **Status:** Won't-fix (out of repo scope).

### 3.7 Maintenance difficulty

#### MAINT-01 — Duplicated backend-availability logic (Medium)

- **File/module:** [`_gravity_benchmark/types.py`](../src/lunaris/surrogate/st_lrps/evaluation/_gravity_benchmark/types.py) `GravityModelCache._load` vs [`core/mc_backend_policy.py`](../src/lunaris/core/mc_backend_policy.py).
- **Description:** "is CUDA available → else fall back" logic is implemented twice with different rigor: the MC policy is explicit and provenance-aware; the benchmark loader is ad-hoc with `print`-only fallback (see SCI-01).
- **Impact:** Two code paths drift; the weaker one is the one feeding benchmark provenance.
- **Proposed fix:** Factor a shared backend-resolution/provenance helper (or have the benchmark consume `mc_backend_policy`-style records). Aligns with SCI-01.
- **Target PR:** #3. **Status:** Open.

#### MAINT-02 — Dependency declarations spread across four files (Medium)

- **File/module:** see DEP-05.
- **Description:** Same root cause as DEP-05, called out under maintenance because the ongoing cost is drift on every dependency change.
- **Impact:** Every dep bump must be mirrored in up to four places.
- **Proposed fix:** Consolidate to `pyproject` + generated constraints (DEP-05).
- **Target PR:** #1. **Status:** Open.

### 3.8 Performance bottlenecks

#### PERF-01 — Heavy import-time cost inflates test/CI startup (Low–Medium)

- **File/module:** test collection (≈24 s to collect 1335 tests); import-time `matplotlib.use("Agg")`, torch, and Numba JIT warmups.
- **Description:** Collection alone is slow because importing test modules pulls heavy stacks. A single serial CI job amplifies this.
- **Impact:** Slow CI feedback.
- **Proposed fix:** Group + parallelize tests (CI-05); audit module-level heavy imports that could be deferred (no behavior change).
- **Target PR:** #2 (grouping). **Status:** Open.

---

## 4. Findings → deliverables / PR mapping

Phase 1 work is split into small, reviewable PRs (per §7). Suggested ordering:

| PR | Theme | Findings addressed |
|---|---|---|
| **#1** | Dependency & environment | DEP-01..06, SCI-04, INST-01, INST-02 (packaging check), DOC-02, MAINT-02 |
| **#2** | CI quality gates | CI-01..05, INST-02 (npm audit), INST-03 (periodic data verify), PERF-01 |
| **#3** | Benchmark provenance | **SCI-01**, SCI-02, SCI-03, MAINT-01 |
| **#4** | Repository cleanup | CLEAN-01, CLEAN-02 (CLEAN-03 = no-op) |
| **#5** | Baseline & smoke infra | DOC-03 (`baseline_manifest.json`), INST-01 (headless smoke test), INST-03 (manifest data hash) |

This file (`docs/BASELINE_AUDIT.md`) is the §1/§7 audit deliverable itself.

---

## 5. Acceptance-criteria tracker (§8)

| Acceptance criterion | Current state | Blocking findings |
|---|---|---|
| Clean install on Ubuntu | Not yet verified for the six profiles | DEP-01, DEP-04, DEP-05 |
| Clean install on Windows | Package imports on Win 11 / Py 3.12 ✅ (profiles not yet defined) | DEP-01, DOC-02 |
| Python 3.10–3.12 CI passes | 3.10/3.11 only | CI-01 |
| Wheel installs in clean env | No build/install job | CI-03 |
| All console entry points work | Wired in `pyproject`; not CI-verified | CI-03 |
| Existing CPU tests pass | 1335 collect cleanly; full green run to be captured | DOC-03 |
| Benchmark records real backend & degree | MC: ✅ ; benchmark: ❌ | **SCI-01**, SCI-02 |
| CPU fallback not labeled as GPU | MC: ✅ ; benchmark: ❌ | **SCI-01** |
| Reference smoke results in machine-readable manifest | Not present | DOC-03 |
| No unexplained physical regression | Pinning-dependent | SCI-04, DEP-04 |
| All changes documented | This audit + companions in progress | DOC-01 |

---

## 6. Explicitly out of scope (§9)

Per the Phase 1 charter, this audit and the follow-up PRs **must not**: change the
ST-LRPS network architecture, add loss functions or ML models, add physics
models, alter spherical-harmonics math, change the propagator integrator, relax
tolerances to "improve" benchmarks, make the direct-force model the default
runtime, perform a broad UI redesign, or develop real GPU SH100/SH200 kernels.
The objective is reproducibility and auditability of existing results — not new
scientific output.

---

*Audit performed at commit `94af9693` on Python 3.12.1 / Windows 11. Update the
**Status** fields as Phase 1 PRs land.*
