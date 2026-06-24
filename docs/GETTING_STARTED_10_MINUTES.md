# Getting Started In 10 Minutes

This guide takes a fresh checkout from install to a verified data set, one
propagated lunar orbit, and a plot output. It keeps ST-LRPS out of the critical
path: the main framework should run with classical spherical-harmonic gravity
only.

## 1. Install

From the repository root:

```bash
python -m pip install -e .
python -c "import lunaris; print(lunaris.__version__)"
```

Use `".[all]"` only when you need the UI, ST-LRPS training/evaluation, report
extras, or development tools:

```bash
python -m pip install -e ".[all]"
```

## 2. Download The Minimal Runtime Data

The minimal preset contains the SPICE kernels and GRAIL gravity model needed for
a default propagation run:

```bash
lunaris-data presets
lunaris-data download --preset minimal
lunaris-data verify --preset minimal --runtime
```

For stricter provenance checks, including SPICE GM constants, use:

```bash
lunaris-data download --preset full-gravity
lunaris-data verify --preset full-gravity --strict --runtime
```

If downloads are stored outside the checkout, set `LUNARIS_DATA_DIR` or pass
`--data-dir` to `lunaris-data`.

## 3. Run One Orbit

This command propagates a short 100 km circular lunar orbit and writes all output
under `outputs/simulations/getting_started`:

```bash
lunaris --hours 2 --alt-km 100 --inc-deg 30 --degree 20 \
  --output-dt-s 120 --make-3d-plots off \
  --out-dir outputs/simulations/getting_started
```

Expected behavior:

- the CLI prints the selected gravity file, force flags, initial state, and
  propagation status;
- `run_config.json` is written under the output directory;
- report plots are written under a run subdirectory when report generation
  succeeds.

## 4. Run The Python Example

The examples use the same public building blocks as the CLI:

```bash
python examples/01_basic_propagation.py
```

The script writes a quick altitude plot to:

```text
outputs/examples/basic_propagation/altitude.png
```

## 5. Next Steps

- Use `examples/02_enable_perturbations.py` to enable third-body, SRP,
  relativity, and k2 tides without introducing surface rasters.
- Use `examples/03_batch_run.py` for a small multi-orbit sweep.
- Use `examples/04_perturbation_budget.py` for a compact acceleration breakdown.
- Use `examples/05_st_lrps_runtime.py --model-dir <run-dir>` only when a trained
  ST-LRPS artifact is available.

ST-LRPS remains an optional advanced subsystem. A missing ST-LRPS artifact must
not block classical propagation, data verification, or the basic examples.
