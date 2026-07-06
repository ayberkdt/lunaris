# Common Layer Cleanup Plan

Status legend: todo / done

Scope: `src/lunaris/common/` plus small cross-layer checks that keep the common
layer dependency-light and single-source-of-truth friendly.

## Current Status

The common layer is in good shape:

- done: `common` has no runtime imports from `physics`, `core`, `analysis`,
  `visualization`, `ui`, or `surrogate`.
- done: `common.__init__` exposes the real lazy submodules:
  `math_utils`, `time_utils`, `batch_defs`, `batch_defs`, `paths`,
  `hashing`, and `lunar_data`.
- done: `C_LIGHT` use in CUDA/batch relativity paths comes from
  `common.constants`.
- done: `math_utils` exposes the advertised helpers (`dot3`, `cross3`,
  `vec3_normalize`, `safe_acos`, `wrap_angle_pi`, `coe_to_rv`) and uses the
  `EPS_*` constants documented in `common.constants`.
- done: `time_utils.US_PER_DAY` is derived from `DAY_S`, and
  `J2000_NOON_US` is derived from `US_PER_DAY`.

## Remaining Work

### R1 - Keep `DAY_S` as the time-scale SSOT

- done: Replace ST-LRPS compute-accounting's local `86_400.0` with
  `common.constants.DAY_S`.
- todo: Treat any future raw `86400` in Python source as suspect unless it is
  prose, a generated dependency file, or a deliberately named derived unit.

### R2 - Keep common tests common-only

- done: `tests/test_common_refactor.py` no longer imports
  `lunaris.batch.engine`.
- done: Core sampling design checks moved to `tests/test_batch_sampling_designs.py`.
- todo: Keep future tests under `test_common_*` limited to `lunaris.common`
  modules and standard-library/numpy assertions.

### R3 - Keep public common surface documented

- done: `docs/PUBLIC_API.md` documents stable common modules and the provisional
  helper modules.
- todo: If a helper graduates from provisional to stable, update this document
  and add a contract test for the promised symbols.

## Verification

Run the focused common checks:

```bash
.venv\Scripts\python.exe -m pytest tests/test_common_refactor.py tests/test_math_utils.py tests/test_time_utils.py tests/test_type_defs.py tests/test_type_defs_contracts.py tests/test_batch_output_grid.py tests/test_batch_sampling_designs.py -q
```

Run import-boundary checks when the dependency is installed:

```bash
.venv\Scripts\python.exe -m lint_imports
```

If `lint_imports` is unavailable in the active environment, install the dev
extra or run the equivalent CI environment before treating the import-linter
contract as verified locally.
